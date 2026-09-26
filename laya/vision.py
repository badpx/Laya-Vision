"""Local image descriptions as text input to an unchanged Laya decision model.

This is a caption bridge, not a native multimodal checkpoint. BLIP generates a
description autoregressively; Laya then answers the questions about that text.
Laya's probabilities are conditional on the description, not calibrated image
probabilities, and its token usage excludes the caption generation stage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import warnings


_MAX_IMAGE_PIXELS = 20_000_000


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError("image input requires Pillow; install 'laya[vision]'") from exc
    return Image, ImageOps


def _check_image(image):
    width, height = image.size
    if width < 1 or height < 1 or width * height > _MAX_IMAGE_PIXELS:
        raise ValueError("image must have between 1 and %d pixels" % _MAX_IMAGE_PIXELS)
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("only a single image is supported; animated or multi-page files are not supported")


def _prepare_image(image):
    Image, ImageOps = _pillow()

    def convert(source):
        _check_image(source)
        oriented = ImageOps.exif_transpose(source)
        try:
            return oriented.convert("RGB")
        finally:
            oriented.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        if isinstance(image, Image.Image):
            return convert(image)
        if not isinstance(image, (str, os.PathLike)):
            raise TypeError("image must be one local file path or a PIL image")
        path = os.fspath(image)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("image path must be a nonempty local path")
        # Never fetch user input. Model weights may be downloaded separately by
        # Hugging Face unless the captioner uses local_files_only=True.
        scheme = urlsplit(path).scheme
        if scheme and not (len(scheme) == 1 and path[1:3] in (":\\", ":/")):
            raise ValueError("image URLs are not supported; use a local file or PIL image")
        path = Path(path)
        if not path.is_file():
            raise ValueError("image must be an existing regular local file")
        with Image.open(path) as source:
            return convert(source)


class BlipCaptioner:
    """Generate a local, deterministic BLIP description of one RGB PIL image.

    Weights and optional libraries load on the first call. Use a local model
    directory and ``local_files_only=True`` for offline operation. The default
    model can download weights on first use; image pixels never go to an API.
    ``device=None`` selects CUDA, then MPS, then CPU. Generation uses greedy
    decoding, evaluation mode, and inference mode.
    """

    generation_mode = "autoregressive"

    def __init__(self, model_id_or_path="Salesforce/blip-image-captioning-base", device=None,
                 revision=None, local_files_only=False, max_new_tokens=48):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(model_id_or_path, (str, os.PathLike)) or not os.fspath(model_id_or_path):
            raise ValueError("model_id_or_path must be a nonempty model ID or local path")
        self.model_id = os.fspath(model_id_or_path)
        self.device = device
        self.revision = revision
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._torch = None
        self._load_lock = threading.Lock()

    def _load(self):
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError as exc:
                raise ImportError("BLIP captioning requires the vision dependencies; install 'laya[vision]'") from exc
            device = self.device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            kwargs = {"local_files_only": self.local_files_only}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            processor = BlipProcessor.from_pretrained(self.model_id, **kwargs)
            model = BlipForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
            model = model.to(device).eval()
            self.device = device
            self._torch = torch
            self._processor = processor
            self._model = model

    def __call__(self, image):
        Image, _ = _pillow()
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise TypeError("BlipCaptioner expects one RGB PIL image")
        _check_image(image)
        self._load()
        with self._torch.inference_mode():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            tokens = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, num_beams=1)
            caption = self._processor.decode(tokens[0], skip_special_tokens=True).strip()
        if not caption:
            raise ValueError("captioner returned an empty description")
        return caption


class VisionAgent:
    """Describe one local image, then call an Agent or Router with the description.

    ``captioner`` may be any callable accepting an RGB PIL image and returning a
    nonempty string. By default it is a lazily loaded :class:`BlipCaptioner`.
    Inputs are a local image path or a PIL image, at most 20 million pixels; EXIF
    orientation is applied and animation/multi-page images are rejected.
    """

    def __init__(self, agent, captioner=None):
        if not callable(getattr(agent, "predict", None)):
            raise TypeError("agent must provide a callable predict method")
        if captioner is not None and not callable(captioner):
            raise TypeError("captioner must be callable")
        self.agent = agent
        self.captioner = captioner if captioner is not None else BlipCaptioner()

    def predict(self, image, questions, *, state=None, **predict_kwargs):
        """Return Laya's result plus an auditable ``vision`` metadata dictionary.

        Optional text/dict/list ``state`` is appended as context after the image
        description, so long context cannot push the image description out of
        the start of Laya's input. Laya can still truncate that combined state
        according to its token budget. ``vision.state`` records the exact text
        submitted to ``agent.predict`` before any Laya hooks or tokenization.
        Questions, hooks, and other keyword arguments pass through unchanged.
        The returned ``usage`` describes only the Laya text stage.
        """
        started = time.perf_counter()
        if not isinstance(questions, dict):
            raise TypeError("questions must be a dict")
        if state is not None and not isinstance(state, (str, dict, list)):
            raise TypeError("state must be text, a dict, a list, or None")
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False) if state is not None else None
        prepared = _prepare_image(image)
        prepared_at = time.perf_counter()
        size = prepared.size
        try:
            caption = self.captioner(prepared)
        finally:
            prepared.close()
        captioned_at = time.perf_counter()
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("captioner must return a nonempty string")
        caption = caption.strip()
        text_state = "Image description: " + caption
        if context:
            text_state += "\n\nAdditional context:\n" + context
        result = self.agent.predict(text_state, questions, **predict_kwargs)
        finished = time.perf_counter()
        if not isinstance(result, dict):
            raise TypeError("agent.predict must return a result dict")
        return {
            **result,
            "vision": {
                "mode": "caption_bridge",
                "caption": caption,
                "model_id": getattr(self.captioner, "model_id", None),
                "state": text_state,
                "probability_scope": "text_surrogate_only",
                "caption_generation": getattr(self.captioner, "generation_mode", "unspecified"),
                "end_to_end_single_pass": False,
                "usage_scope": "laya_text_stage_only",
                "image": {"width": size[0], "height": size[1], "mode": "RGB"},
                "timings_ms": {
                    "image_prepare": (prepared_at - started) * 1000.0,
                    "caption": (captioned_at - prepared_at) * 1000.0,
                    "laya": (finished - captioned_at) * 1000.0,
                    "total": (finished - started) * 1000.0,
                },
            },
        }
