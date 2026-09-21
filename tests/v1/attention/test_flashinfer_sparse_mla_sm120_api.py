# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_bf16,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
    _required_sm120_sparse_topk,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_deepseek_v4_sm120_pads_eight_query_heads() -> None:
    assert DeepseekV4FlashInferSM120Attention.get_padded_num_q_heads(8) == 16
    assert DeepseekV4FlashInferSM120Attention.get_padded_num_q_heads(32) == 32


def test_deepseek_v4_o_proj_supports_dequantized_weights() -> None:
    weight = torch.arange(48, dtype=torch.bfloat16).reshape(3, 16) / 64

    class QuantizedLinear:
        weight = None

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(x, weight)

    wo_a = QuantizedLinear()
    o = torch.arange(32, dtype=torch.bfloat16).reshape(2, 2, 8) / 16
    positions = torch.tensor([0, 1])
    cos_sin = torch.tensor([[0.8, 0.6], [0.5, -0.25]])

    actual = deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin,
        wo_a,
        lambda x: x,
        n_groups=1,
        heads_per_group=2,
        nope_dim=6,
        rope_dim=2,
        o_lora_rank=3,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )

    expected_o = o.float().clone()
    cos = cos_sin[positions, 0].unsqueeze(-1)
    sin = cos_sin[positions, 1].unsqueeze(-1)
    even = o[..., 6].float()
    odd = o[..., 7].float()
    expected_o[..., 6] = even * cos + odd * sin
    expected_o[..., 7] = odd * cos - even * sin
    expected = expected_o.to(torch.bfloat16).flatten(1) @ weight.T
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_inv_rope_bf16_matches_reference() -> None:
    tokens, groups, heads_per_group = 17, 2, 4
    nope_dim, rope_dim = 448, 64
    generator = torch.Generator(device="cuda").manual_seed(7)
    o = torch.randn(
        tokens,
        groups * heads_per_group,
        nope_dim + rope_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(tokens, dtype=torch.long, device="cuda")
    cos_sin = torch.randn(
        tokens,
        rope_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    actual = fused_inv_rope_bf16(
        o,
        positions,
        cos_sin,
        n_groups=groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )

    rope = o[..., nope_dim:]
    cos = cos_sin[positions, : rope_dim // 2].unsqueeze(-2)
    sin = cos_sin[positions, rope_dim // 2 :].unsqueeze(-2)
    even = rope[..., 0::2].float()
    odd = rope[..., 1::2].float()
    inv_rope = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1
    ).flatten(-2)
    expected = torch.cat((o[..., :nope_dim].float(), inv_rope), dim=-1)
    expected = expected.to(o.dtype).reshape(
        tokens, groups, heads_per_group * (nope_dim + rope_dim)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192
