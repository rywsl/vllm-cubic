# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any, cast

from PIL import Image
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.inputs import VisionChunk
from vllm.tokenizers.hf import HfTokenizer


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for Kimi-K3 image and video chunks.

    K3 exposes the standard ``image`` modality, so vLLM calls this processor
    with ``images=[PIL, ...]``. The underlying checkpoint image processor
    (``KimiK3VisionProcessor``) works on ``{"type": "image", "image": PIL}``
    media dicts, so this wrapper adapts bare PIL images into that shape before
    delegating to ``preprocess``.

    Text is only tokenized here; the single ``<|kimi_image_placeholder|>``
    token per image is expanded into the resolution-aware media block by the
    model's ``_get_prompt_updates`` on the vLLM side.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    @staticmethod
    def _to_pil_frame(frame: object) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if hasattr(frame, "media"):
            frame = frame.media
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame_any: Any = frame
        if getattr(frame_any, "ndim", None) is None:
            raise TypeError(f"Unsupported video frame type: {type(frame)}")
        if frame_any.ndim != 3:
            raise ValueError(f"Video frames must be 3D, got shape {frame_any.shape}")
        # vLLM's video loaders return RGB HWC arrays. Accept CHW tensors too.
        if frame_any.shape[0] in (1, 3, 4) and frame_any.shape[-1] not in (1, 3, 4):
            frame_any = frame_any.transpose(1, 2, 0)
        if frame_any.shape[-1] == 1:
            frame_any = frame_any[..., 0]
        if str(frame_any.dtype).startswith("float"):
            frame_any = (frame_any.clip(0, 1) * 255).astype("uint8")
        else:
            frame_any = frame_any.astype("uint8", copy=False)
        return Image.fromarray(frame_any).convert("RGB")

    @classmethod
    def _as_media(cls, chunk: object) -> dict[str, Any]:
        """Normalize vLLM's decoded video tuple into a vision-chunk dict."""
        if hasattr(chunk, "media"):
            chunk = chunk.media
        if isinstance(chunk, dict):
            if chunk.get("type") == "image":
                return chunk
            if chunk.get("type") == "video_chunk":
                frames = cast(Any, chunk.get("video_chunk"))
                if frames is None:
                    raise ValueError("Video input must contain frames")
                return {
                    "type": "video_chunk",
                    "video_chunk": [cls._to_pil_frame(f) for f in frames],
                }
        if isinstance(chunk, tuple) and len(chunk) == 2:
            chunk = chunk[0]
        if isinstance(chunk, Image.Image):
            return {"type": "image", "image": chunk.convert("RGB")}
        if isinstance(chunk, Sequence) and not isinstance(chunk, (str, bytes)):
            frames = [cls._to_pil_frame(frame) for frame in chunk]
            if not frames:
                raise ValueError("Video input must contain at least one frame")
            return {"type": "video_chunk", "video_chunk": frames}
        # A decoded ndarray/tensor has a leading temporal dimension.
        chunk_any: Any = chunk
        shape = getattr(chunk_any, "shape", None)
        if shape is not None and len(shape) == 4:
            frames = [cls._to_pil_frame(frame) for frame in chunk_any]
            return {"type": "video_chunk", "video_chunk": frames}
        raise TypeError(f"Unsupported Kimi-K3 vision input: {type(chunk)}")

    def _limit_video_frames(self, media: dict[str, Any]) -> dict[str, Any]:
        """Keep frame stacks within K3's temporal position-embedding range."""
        if media.get("type") != "video_chunk":
            return media
        frames = media["video_chunk"]
        max_frames = getattr(self.image_processor, "num_frames_per_chunk", 4)
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

    def __call__(
        self,
        text: str | list[str] | None = None,
        vision_chunks: list[VisionChunk] | None = None,
        images: object | list[object] | None = None,
        videos: object | list[object] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if vision_chunks is not None:
            medias = [
                self._limit_video_frames(self._as_media(chunk))
                for chunk in vision_chunks
            ]
        else:
            medias = []
            if images is not None:
                image_list = images if isinstance(images, list) else [images]
                medias.extend(self._as_media(image) for image in image_list)
            if videos is not None:
                video_list = videos if isinstance(videos, list) else [videos]
                medias.extend(
                    self._limit_video_frames(self._as_media(video))
                    for video in video_list
                )

        if medias:
            mm_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
            )
        else:
            mm_inputs = {}

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **mm_inputs},
            tensor_type=return_tensors,
        )
