#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

KIMI_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
KIMI_PYTHON="$KIMI_ROOT/.venv/bin/python"
KIMI_DRY_RUN=0
KIMI_COMMAND=
KIMI_PROFILE=baseline

usage() {
    cat <<'EOF'
Usage: kimi_k3_h200.sh {build|serve|gpu-tests} [PROFILE] [--dry-run]

Profiles: baseline, cache32, dcp2, dcp4, dcp8, dspark, recoverssm
RecoverSSM adds recovery to dSpark with model runner V2. Other profiles use V1.
All serving profiles use TP8, EP, PP1, BF16 activations, and align prefix caching.

Overrides (environment variables):
  KIMI_MODEL / KIMI_REVISION       Target HF identifier or local model path / SHA
  KIMI_TOKENIZER_REVISION         Defaults to the target revision
  KIMI_MAX_MODEL_LEN=1048576      KIMI_MAX_NUM_SEQS=128
  KIMI_BATCHED_TOKENS=2048        KIMI_GPU_MEMORY_UTILIZATION=0.95
  KIMI_KV_CACHE_DTYPE=fp8_q16     auto, bfloat16, fp8_q16, or cubic8
  KIMI_DYNAMIC_A8=1              KIMI_SEED=42
  KIMI_API_COMPAT=1              Enable the K3 API contract used by KVV
  KIMI_HOST=127.0.0.1            KIMI_PORT=8000
  KIMI_MANIFEST=artifacts/kimi-k3-server.json
  KIMI_TEST_REPORT=artifacts/kimi-k3-h200-gpu-tests.xml
  MAX_JOBS=8                    NVCC_THREADS=2

build compiles this checkout's CUDA extensions, then installs focused test tools.
gpu-tests runs kernel/loading/cache checks; model quality and serving performance
require the separate model evaluation and benchmark workflow.
--dry-run validates arguments and prints commands without executing them.
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
for kimi_arg in "$@"; do
    case "$kimi_arg" in
        --help|-h) usage; exit 0 ;;
        --dry-run) KIMI_DRY_RUN=1 ;;
        build|serve|gpu-tests)
            [[ -z "$KIMI_COMMAND" ]] || fail 'Specify one subcommand'
            KIMI_COMMAND=$kimi_arg ;;
        baseline|cache32|dcp2|dcp4|dcp8|dspark|recoverssm)
            [[ ${kimi_profile_seen:-0} == 0 ]] || fail 'Specify one profile'
            KIMI_PROFILE=$kimi_arg
            kimi_profile_seen=1 ;;
        *) fail "Unknown argument: $kimi_arg" ;;
    esac
done
[[ -n "$KIMI_COMMAND" ]] || { usage >&2; exit 2; }
if [[ "$KIMI_COMMAND" == build && ${kimi_profile_seen:-0} == 1 ]]; then
    fail 'build does not take a serving profile'
fi

run() {
    printf '%q ' "$@"
    printf '\n'
    if [[ "$KIMI_DRY_RUN" == 0 ]]; then "$@"; fi
}

positive_integer() {
    local name=$1 value=$2
    [[ "$value" =~ ^[1-9][0-9]{0,8}$ ]] || fail "$name must be a positive integer"
}

KIMI_MODEL=${KIMI_MODEL:-QuantTrio/Kimi-K3-Cubic-2.5Bit}
KIMI_REVISION=${KIMI_REVISION:-f29f15dc4afd99feb3349b538bbe7ed439787853}
KIMI_TOKENIZER_REVISION=${KIMI_TOKENIZER_REVISION:-$KIMI_REVISION}
KIMI_DRAFT_MODEL=Inferact/Kimi-K3-DSpark
KIMI_DRAFT_REVISION=cf6b8244620e7ea4b0651d214f28e89eac75bed6
KIMI_MAX_MODEL_LEN=${KIMI_MAX_MODEL_LEN:-1048576}
KIMI_MAX_NUM_SEQS=${KIMI_MAX_NUM_SEQS:-128}
KIMI_BATCHED_TOKENS=${KIMI_BATCHED_TOKENS:-2048}
KIMI_GPU_MEMORY_UTILIZATION=${KIMI_GPU_MEMORY_UTILIZATION:-0.95}
KIMI_KV_CACHE_DTYPE=${KIMI_KV_CACHE_DTYPE:-fp8_q16}
KIMI_DYNAMIC_A8=${KIMI_DYNAMIC_A8:-1}
KIMI_API_COMPAT=${KIMI_API_COMPAT:-1}
KIMI_SEED=${KIMI_SEED:-42}
KIMI_HOST=${KIMI_HOST:-127.0.0.1}
KIMI_PORT=${KIMI_PORT:-8000}
KIMI_MANIFEST=${KIMI_MANIFEST:-$KIMI_ROOT/artifacts/kimi-k3-server.json}
KIMI_TEST_REPORT=${KIMI_TEST_REPORT:-$KIMI_ROOT/artifacts/kimi-k3-h200-gpu-tests.xml}
positive_integer KIMI_MAX_MODEL_LEN "$KIMI_MAX_MODEL_LEN"
positive_integer KIMI_MAX_NUM_SEQS "$KIMI_MAX_NUM_SEQS"
positive_integer KIMI_BATCHED_TOKENS "$KIMI_BATCHED_TOKENS"
positive_integer KIMI_PORT "$KIMI_PORT"
(( KIMI_PORT <= 65535 )) || fail 'KIMI_PORT must be at most 65535'
[[ "$KIMI_GPU_MEMORY_UTILIZATION" =~ ^0\.0*[1-9][0-9]*$|^1(\.0+)?$ ]] ||
    fail 'KIMI_GPU_MEMORY_UTILIZATION must be in (0, 1]'
[[ "$KIMI_DYNAMIC_A8" =~ ^[01]$ ]] || fail 'KIMI_DYNAMIC_A8 must be 0 or 1'
[[ "$KIMI_API_COMPAT" =~ ^[01]$ ]] || fail 'KIMI_API_COMPAT must be 0 or 1'
[[ "$KIMI_SEED" =~ ^(0|[1-9][0-9]{0,8})$ ]] || fail 'KIMI_SEED must be nonnegative'
[[ "$KIMI_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail 'KIMI_REVISION must be a full commit SHA'
[[ "$KIMI_TOKENIZER_REVISION" =~ ^[0-9a-f]{40}$ ]] ||
    fail 'KIMI_TOKENIZER_REVISION must be a full commit SHA'
case "$KIMI_KV_CACHE_DTYPE" in
    auto|bfloat16|fp8_q16|cubic8) ;;
    *) fail 'Unsupported KIMI_KV_CACHE_DTYPE' ;;
esac
if [[ "$KIMI_PROFILE" == dspark || "$KIMI_PROFILE" == recoverssm ]]; then
    [[ "$KIMI_KV_CACHE_DTYPE" == fp8_q16 ]] ||
        fail 'dSpark experiments require the fp8_q16 target cache'
fi

cd -- "$KIMI_ROOT"
if [[ "$KIMI_COMMAND" == build ]]; then
    positive_integer MAX_JOBS "${MAX_JOBS:-8}"
    positive_integer NVCC_THREADS "${NVCC_THREADS:-2}"
    if [[ "$KIMI_DRY_RUN" == 0 ]]; then
        command -v uv >/dev/null || fail 'Install uv before building'
        command -v nvcc >/dev/null || fail 'Install the CUDA toolkit and expose nvcc on PATH'
    fi
    if [[ ! -x "$KIMI_PYTHON" ]]; then
        run uv venv --python 3.12 "$KIMI_ROOT/.venv"
    fi
    run "$KIMI_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 .venv"'
    run uv pip install --python "$KIMI_PYTHON" -r requirements/build/cuda.txt --torch-backend=auto
    run env -u VLLM_PRECOMPILED_WHEEL_LOCATION VLLM_USE_PRECOMPILED=0 \
        VLLM_USE_PRECOMPILED_RUST=0 VLLM_TARGET_DEVICE=cuda \
        TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-2}" \
        uv pip install --python "$KIMI_PYTHON" --no-build-isolation -e . --torch-backend=auto
    run uv pip install --python "$KIMI_PYTHON" pytest pytest-asyncio pytest-forked pytest-timeout tblib scipy
    exit 0
fi

if [[ "$KIMI_DRY_RUN" == 0 && ! -x "$KIMI_PYTHON" ]]; then
    fail 'Build the checkout first: kimi_k3_h200.sh build'
fi
run "$KIMI_PYTHON" -c '
import torch
import vllm._C
import vllm._flashmla_extension_C
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 8, "Expose exactly eight H200 GPUs"
for i in range(8):
    assert "H200" in torch.cuda.get_device_name(i), "This package targets 8 x H200"
assert hasattr(torch.ops._C, "cubic_w2_a8_gemv"), "Build the Cubic CUDA extensions"
assert hasattr(torch.ops._flashmla_extension_C, "fwd_kvcache_mla_fp8_q16"), "Build patched FlashMLA"
'

kimi_runner=0
[[ "$KIMI_PROFILE" != recoverssm ]] || kimi_runner=1
kimi_env=(VLLM_CUBIC_DYNAMIC_A8="$KIMI_DYNAMIC_A8"
    VLLM_USE_V2_MODEL_RUNNER="$kimi_runner" VLLM_SERVER_DEV_MODE=1
    VLLM_KIMI_K3_API_COMPAT="$KIMI_API_COMPAT" VLLM_ENFORCE_STRICT_TOOL_CALLING=1)
if [[ "$KIMI_COMMAND" == gpu-tests ]]; then
    run mkdir -p -- "$(dirname -- "$KIMI_TEST_REPORT")"
    run env "${kimi_env[@]}" "$KIMI_PYTHON" -m pytest -q \
        tests/quantization/test_cubic.py tests/quantization/test_cubic_dynamic_a8.py \
        tests/quantization/test_cubic_warmup.py \
        tests/kernels/quantization/test_marlin_tile_padding.py \
        tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py \
        tests/kernels/mamba/test_gdn_fused_mtp.py \
        tests/kernels/mamba/test_precopy_mamba_align.py \
        tests/v1/test_cubic8_mla_kv.py tests/v1/test_fp8_q16.py \
        tests/models/kimi_k3/test_kda.py tests/models/kimi_k3/test_weight_loading.py \
        tests/models/kimi_k3/test_kda_metadata.py \
        tests/models/kimi_k3/test_mla_prefill_context.py \
        tests/v1/core/test_mamba_align_chunk_split.py \
        tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
        tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
        tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py \
        --junitxml="$KIMI_TEST_REPORT"
    exit 0
fi

kimi_serve=("$KIMI_PYTHON" -m vllm.entrypoints.cli.main serve "$KIMI_MODEL"
    --revision "$KIMI_REVISION" --tokenizer-revision "$KIMI_TOKENIZER_REVISION"
    --trust-remote-code --code-revision "$KIMI_REVISION"
    --served-model-name Kimi-K3-Cubic-2.5Bit --dtype bfloat16 --quantization cubic
    --reasoning-parser kimi_k3 --enable-auto-tool-choice --tool-call-parser kimi_k3
    --tensor-parallel-size 8 --enable-expert-parallel --pipeline-parallel-size 1
    --kv-cache-dtype "$KIMI_KV_CACHE_DTYPE" --attention-backend FLASHMLA
    --enable-prefix-caching
    --mamba-cache-mode align --mamba-backend TRITON --enable-chunked-prefill
    --max-model-len "$KIMI_MAX_MODEL_LEN" --max-num-seqs "$KIMI_MAX_NUM_SEQS"
    --max-num-batched-tokens "$KIMI_BATCHED_TOKENS"
    --gpu-memory-utilization "$KIMI_GPU_MEMORY_UTILIZATION" --seed "$KIMI_SEED"
    --host "$KIMI_HOST" --port "$KIMI_PORT")
case "$KIMI_PROFILE" in
    baseline) kimi_serve+=(--decode-context-parallel-size 1) ;;
    cache32) kimi_serve+=(--decode-context-parallel-size 1 --prefix-match-unit 32) ;;
    dcp2|dcp4|dcp8) kimi_serve+=(--decode-context-parallel-size "${KIMI_PROFILE#dcp}") ;;
    dspark|recoverssm)
        kimi_spec='{"method":"dspark","model":"'"$KIMI_DRAFT_MODEL"'","revision":"'"$KIMI_DRAFT_REVISION"'","code_revision":"'"$KIMI_DRAFT_REVISION"'","num_speculative_tokens":7,"quantization":null,"kv_cache_dtype":"auto","attention_backend":"TRITON_MLA"}'
        kimi_serve+=(--decode-context-parallel-size 1 --speculative-config "$kimi_spec")
        if [[ "$KIMI_PROFILE" == recoverssm ]]; then
            kimi_serve+=(--use-replayssm --mamba-ssm-cache-dtype float32)
        fi ;;
esac

run "$KIMI_PYTHON" - "$KIMI_MANIFEST" "$KIMI_PROFILE" "${kimi_env[@]}" -- "${kimi_serve[@]}" <<'PY'
import datetime
import json
import os
import pathlib
import subprocess
import sys

import torch

path, profile = sys.argv[1:3]
separator = sys.argv.index("--", 3)
environment = {
    key: value for key, value in os.environ.items()
    if key.startswith("VLLM_CUBIC_") or key in {
        "CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCL_PROTO", "NCCL_NVLS_ENABLE",
        "NCCL_P2P_DISABLE", "NCCL_NET_GDR_LEVEL", "VLLM_BATCH_INVARIANT",
        "VLLM_USE_V2_MODEL_RUNNER", "VLLM_GDN_DECODE_KERNEL",
        "VLLM_KIMI_K3_API_COMPAT", "VLLM_ENFORCE_STRICT_TOOL_CALLING",
    }
}
environment.update(value.split("=", 1) for value in sys.argv[3:separator])
argv = sys.argv[separator + 1:]
def option(name):
    return argv[argv.index(name) + 1] if name in argv else None
spec = json.loads(option("--speculative-config") or "{}")
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "profile": profile,
    "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
    "model": argv[argv.index("serve") + 1],
    "model_revision": option("--revision"),
    "code_revision": option("--code-revision"),
    "tokenizer_revision": option("--tokenizer-revision"),
    "draft_model": spec.get("model"),
    "draft_revision": spec.get("revision"),
    "draft_code_revision": spec.get("code_revision"),
    "environment": environment,
    "argv": argv,
    "gpu_devices": [
        {"name": torch.cuda.get_device_name(i),
         "capability": list(torch.cuda.get_device_capability(i)),
         "total_memory": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ],
}
destination = pathlib.Path(path)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(manifest, indent=2) + "\n")
PY

if [[ "$KIMI_DRY_RUN" == 1 ]]; then
    run env "${kimi_env[@]}" "${kimi_serve[@]}"
else
    exec env "${kimi_env[@]}" "${kimi_serve[@]}"
fi
