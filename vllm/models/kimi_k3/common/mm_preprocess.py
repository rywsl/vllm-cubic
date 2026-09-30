# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from transformers import BatchFeature

from vllm.config.multimodal import ImageDummyOptions, MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    VisionChunkProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)
from vllm.utils.import_utils import is_numba_available

logger = init_logger(__name__)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K3 image and video chunks.

    Both images and decoded video frame chunks use the unified
    ``vision_chunk`` modality. This keeps the flattened ``pixel_values`` and
    ``grid_thws`` fields aligned when a request mixes images and videos.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        processor_cls = KimiK25FusedVisionProcessor if is_numba_available() else None
        if processor_cls is None:
            raise RuntimeError("Kimi-K3 video support requires numba for preprocessing")
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls_overrides=processor_cls,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"vision_chunk": None}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds image-based dummy inputs for K3 profiling.

    The dummy text is made of ``<|kimi_image_placeholder|>`` tokens — exactly
    the placeholder that K3's ``_get_prompt_updates`` expands — and the dummy
    mm data is a plain list of PIL images under the ``image`` key.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_media = mm_counts.get("vision_chunk", 0)
        return self.info.get_hf_config().image_placeholder * num_media

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("vision_chunk") if mm_options else None,
        )
        return {
            "vision_chunk": [
                {"type": "image", "image": image}
                for image in self._get_dummy_images(
                    width=max_size.width,
                    height=max_size.height,
                    num_images=mm_counts.get("vision_chunk", 0),
                    overrides=image_overrides,
                )
            ]
        }


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Multi-modal processor for Kimi-K3 images and decoded video frames."""

    @staticmethod
    def _get_media_size(media: dict[str, Any]) -> tuple[int, int]:
        frame = media["image"] if media["type"] == "image" else media["video_chunk"][0]
        if hasattr(frame, "media"):
            frame = frame.media
        if hasattr(frame, "size") and not hasattr(frame, "shape"):
            width, height = frame.size
            return int(width), int(height)
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError(f"Unsupported media frame shape: {shape}")
        if shape[-1] in (1, 3, 4):
            height, width = shape[:2]
        else:
            height, width = shape[-2:]
        return int(width), int(height)

    def _limit_video_frames(self, media: dict[str, Any]) -> dict[str, Any]:
        if media.get("type") != "video_chunk":
            return media
        frames = media["video_chunk"]
        max_frames = getattr(self.info.image_processor, "num_frames_per_chunk", 4)
        if len(frames) <= max_frames:
            return media
        if max_frames <= 1:
            indices = [0]
        else:
            indices = [
                round(i * (len(frames) - 1) / (max_frames - 1))
                for i in range(max_frames)
            ]
        return {**media, "video_chunk": [frames[i] for i in indices]}

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensor back into per-image items.

        ``pixel_values`` holds all patches from every image concatenated; each
        image's patch count is ``prod(grid_thws[i])``. ``grid_thws`` is one
        ``[N_t, N_h, N_w]`` row per image.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 media placeholder into a resolution-aware update.

        K3's prompt carries a single ``<|kimi_image_placeholder|>`` token per
        image. This replaces that token with
        ``<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>``,
        embedding the per-image resolution in the prompt and marking only the
        ``<|media_pad|>`` positions as embedding slots (the number of pads is
        the feature size returned by ``media_tokens_calculator``).
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            media_items = mm_items.get_items("vision_chunk", VisionChunkProcessorItems)
            media = media_items.get(item_idx)
            if media is None:
                raise ValueError(f"Missing media data at index {item_idx}")
            if hasattr(media, "media"):
                media = media.media
            if isinstance(media, dict):
                media_dict = media
            elif isinstance(media, tuple) and len(media) == 2:
                media_dict = {"type": "video_chunk", "video_chunk": media[0]}
            else:
                media_dict = {"type": "image", "image": media}
            media_dict = self._limit_video_frames(media_dict)
            num_media_token = self.info.media_tokens_calculator(media_dict)
            pads = media_token * num_media_token

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not the
            # post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor
            # (`KimiK3VisionProcessor.preprocess`) only produces new tensors via
            # `image.resize(...)` and never mutates the stored PIL. This matches
            # the reference HF processor (`KimiK3Processor.preprocess_medias`),
            # which also builds the prompt from the original `img.size`. The
            # resize is reflected only in the pad count above.
            width, height = self._get_media_size(media_dict)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, full, add_special_tokens=False),
                media_token_id,
            )

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=cached_encode(
                    tokenizer, image_placeholder, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
        ]
