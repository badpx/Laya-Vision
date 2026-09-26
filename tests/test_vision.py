"""Offline contract and input-boundary tests for the optional local image bridge.

Run: python -m pytest tests/test_vision.py
No checkpoint downloads or real model inference are needed by this suite.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Image = pytest.importorskip("PIL.Image")

from laya import BlipCaptioner, VisionAgent  # noqa: E402
import laya.vision as vision  # noqa: E402


QUESTIONS = {
    "scene": {"type": "choice", "criteria": ["cat", "dog"]},
    "cat_present": {"type": "noul", "instructions": "There is a cat."},
    "quality": {"type": "score", "criteria": ["poor", "good", "excellent"]},
}


class Recorder:
    def __init__(self):
        self.calls = []
        self.result = {
            "model": "fake-laya",
            "answers": {
                "scene": {"type": "choice", "choice": "cat", "probabilities": {"cat": 0.9, "dog": 0.1}},
                "cat_present": {"type": "noul", "noul": 0.9},
                "quality": {"type": "score", "score": 0.8},
            },
            "usage": {"input_tokens": 20, "output_tokens": 0},
            "routing": {"model": "english"},
        }

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self.result


class Captioner:
    model_id = "offline-test-captioner"

    def __init__(self, caption="a cat sitting on a chair"):
        self.caption = caption
        self.calls = []

    def __call__(self, image):
        self.calls.append({"mode": image.mode, "size": image.size, "pixel": image.getpixel((0, 0))})
        return self.caption


def test_import_and_construction_do_not_load_optional_libraries():
    code = """
import sys
import laya
from laya import BlipCaptioner, VisionAgent
captioner = BlipCaptioner(local_files_only=True)
agent = VisionAgent(type('Agent', (), {'predict': lambda *args: {}})())
assert captioner._model is None
assert agent.captioner._model is None
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'PIL' not in sys.modules
assert 'VisionAgent' in laya.__all__ and 'BlipCaptioner' in laya.__all__
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)


def test_result_schema_passthrough_and_audit_state():
    agent, captioner = Recorder(), Captioner()
    hook = object()
    kwargs = {"hooks": [hook], "max_len": 128, "head_max_len": 16, "model": "english", "lang": "en"}
    with Image.new("RGB", (4, 3), "red") as image:
        result = VisionAgent(agent, captioner).predict(image, QUESTIONS, state={"context": "中文"}, **kwargs)
        assert image.getpixel((0, 0)) == (255, 0, 0)
    state, questions, forwarded = agent.calls[0]
    assert questions is QUESTIONS
    assert forwarded == kwargs
    assert state == 'Image description: a cat sitting on a chair\n\nAdditional context:\n{"context": "中文"}'
    assert result["answers"] is agent.result["answers"]
    assert result["usage"] is agent.result["usage"]
    assert result["routing"] is agent.result["routing"]
    assert result is not agent.result
    assert "vision" not in agent.result
    metadata = result["vision"]
    assert metadata["state"] == state
    assert metadata["caption"] == captioner.caption
    assert metadata["model_id"] == captioner.model_id
    assert metadata["mode"] == "caption_bridge"
    assert metadata["probability_scope"] == "text_surrogate_only"
    assert metadata["end_to_end_single_pass"] is False
    assert metadata["usage_scope"] == "laya_text_stage_only"
    assert metadata["image"] == {"width": 4, "height": 3, "mode": "RGB"}
    timings = metadata["timings_ms"]
    assert set(timings) == {"image_prepare", "caption", "laya", "total"}
    assert all(value >= 0 for value in timings.values())
    assert timings["total"] == pytest.approx(sum(timings[key] for key in ("image_prepare", "caption", "laya")))
    json.dumps(result)


@pytest.mark.parametrize("state", [None, "", "a" * 10000, ["context", "中文"]])
def test_caption_precedes_any_context(state):
    agent = Recorder()
    with Image.new("RGB", (1, 1)) as image:
        result = VisionAgent(agent, Captioner("  a cat  ")).predict(image, QUESTIONS, state=state)
    actual = agent.calls[0][0]
    assert actual.startswith("Image description: a cat")
    assert result["vision"]["state"] == actual
    if state:
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        assert actual.endswith(context)
    else:
        assert actual == "Image description: a cat"


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_rgb_conversion_does_not_close_or_modify_caller_image(mode):
    captioner = Captioner()
    with Image.new(mode, (4, 3)) as image:
        VisionAgent(Recorder(), captioner).predict(image, QUESTIONS)
        assert image.mode == mode
        assert image.size == (4, 3)
        image.getpixel((0, 0))
    assert captioner.calls[0]["mode"] == "RGB"


def test_local_path_exif_orientation_and_no_network(tmp_path, monkeypatch):
    path = tmp_path / "rotated.jpg"
    with Image.new("RGB", (2, 3), "red") as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)

    def no_network(*args, **kwargs):
        raise AssertionError("an offline captioner must never trigger network access")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    captioner = Captioner()
    result = VisionAgent(Recorder(), captioner).predict(path, QUESTIONS)
    assert captioner.calls[0]["size"] == (3, 2)
    assert result["vision"]["image"] == {"width": 3, "height": 2, "mode": "RGB"}
    with Image.open(path) as original:
        assert original.size == (2, 3)
        assert original.getexif()[274] == 6


@pytest.mark.parametrize("image", [None, [], ["a.png", "b.png"], b"image", 42, "",
                                    "https://example.com/private.png", "http://example.com/a.png",
                                    "file:///tmp/a.png", "data:image/png;base64,AA=="])
def test_invalid_or_missing_image_never_calls_either_model(image):
    agent, captioner = Recorder(), Captioner()
    with pytest.raises((TypeError, ValueError)):
        VisionAgent(agent, captioner).predict(image, QUESTIONS)
    assert not agent.calls
    assert not captioner.calls


def test_default_captioner_does_not_load_for_missing_file(tmp_path, monkeypatch):
    def no_load(self):
        raise AssertionError("BLIP must not load for an invalid input")

    monkeypatch.setattr(BlipCaptioner, "_load", no_load)
    bridge = VisionAgent(Recorder())
    with pytest.raises(ValueError, match="regular local file"):
        bridge.predict(tmp_path / "missing.jpg", QUESTIONS)
    assert bridge.captioner._model is None
    assert not bridge.agent.calls


@pytest.mark.parametrize("kind", ["directory", "corrupt", "animation", "oversized", "decompression_bomb"])
def test_invalid_file_boundaries_precede_model_calls(kind, tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    if kind == "directory":
        path.mkdir()
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "animation":
        path = tmp_path / "image.gif"
        with Image.new("RGB", (2, 2), "red") as first, Image.new("RGB", (2, 2), "blue") as second:
            first.save(path, save_all=True, append_images=[second], duration=10, loop=0)
    else:
        with Image.new("RGB", (10, 3)) as image:
            image.save(path)
        if kind == "oversized":
            monkeypatch.setattr(vision, "_MAX_IMAGE_PIXELS", 20)
        else:
            monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20)
    agent, captioner = Recorder(), Captioner()
    with pytest.raises((ValueError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError)):
        VisionAgent(agent, captioner).predict(path, QUESTIONS)
    assert not agent.calls
    assert not captioner.calls


@pytest.mark.parametrize("caption", ["", " \n", None, 1, ["cat"]])
def test_invalid_caption_never_reaches_laya(caption):
    agent = Recorder()
    with Image.new("RGB", (1, 1)) as image, pytest.raises(ValueError, match="nonempty string"):
        VisionAgent(agent, Captioner(caption)).predict(image, QUESTIONS)
    assert not agent.calls


def test_input_and_dependency_errors_are_not_silently_replaced():
    agent, captioner = Recorder(), Captioner()
    with Image.new("RGB", (1, 1)) as image:
        for kwargs in ({"state": 5}, {"state": {"bad": object()}}):
            with pytest.raises(TypeError):
                VisionAgent(agent, captioner).predict(image, QUESTIONS, **kwargs)
        with pytest.raises(TypeError, match="questions"):
            VisionAgent(agent, captioner).predict(image, [])
        assert not captioner.calls

        def broken_captioner(image):
            raise RuntimeError("caption failed")

        with pytest.raises(RuntimeError, match="caption failed"):
            VisionAgent(agent, broken_captioner).predict(image, QUESTIONS)
        assert not agent.calls

        def broken_predict(*args, **kwargs):
            raise RuntimeError("decision failed")

        agent.predict = broken_predict
        with pytest.raises(RuntimeError, match="decision failed"):
            VisionAgent(agent, captioner).predict(image, QUESTIONS)
        assert image.getpixel((0, 0)) == (0, 0, 0)


def test_invalid_constructors_and_result():
    with pytest.raises(TypeError, match="predict"):
        VisionAgent(object())
    with pytest.raises(TypeError, match="captioner"):
        VisionAgent(Recorder(), captioner="not callable")
    for value in (0, -1, True, 1.5, "48"):
        with pytest.raises(ValueError, match="positive integer"):
            BlipCaptioner(max_new_tokens=value)
    with pytest.raises(ValueError, match="model ID"):
        BlipCaptioner(model_id_or_path="")
    with Image.new("RGB", (1, 1)) as image, pytest.raises(TypeError, match="result dict"):
        VisionAgent(SimpleNamespace(predict=lambda *args: None), Captioner()).predict(image, QUESTIONS)


def test_blip_loading_and_greedy_inference_with_local_weights(monkeypatch):
    events = []
    inside_inference = False

    @contextmanager
    def inference_mode():
        nonlocal inside_inference
        inside_inference = True
        try:
            yield
        finally:
            inside_inference = False

    class Batch(dict):
        def to(self, device):
            events.append(("batch.to", device))
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            events.append(("processor.load", model_id, kwargs))
            return cls()

        def __call__(self, images, return_tensors):
            assert inside_inference
            assert images.mode == "RGB"
            assert return_tensors == "pt"
            return Batch(pixel_values="local-pixels")

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [1, 2, 3]
            assert skip_special_tokens
            return " a cat "

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            events.append(("model.load", model_id, kwargs))
            return cls()

        def to(self, device):
            events.append(("model.to", device))
            return self

        def eval(self):
            events.append(("model.eval",))
            return self

        def generate(self, **kwargs):
            assert inside_inference
            assert ("model.eval",) in events
            assert kwargs == {"pixel_values": "local-pixels", "max_new_tokens": 12,
                              "do_sample": False, "num_beams": 1}
            events.append(("generate",))
            return [[1, 2, 3]]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=inference_mode))
    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(BlipProcessor=Processor, BlipForConditionalGeneration=Model))
    captioner = BlipCaptioner("/local/blip", device="cpu", revision="pinned", local_files_only=True,
                              max_new_tokens=12)
    assert events == []
    with Image.new("RGB", (1, 1)) as image:
        result = VisionAgent(Recorder(), captioner).predict(image, QUESTIONS)
        assert captioner(image) == "a cat"
    assert result["vision"]["caption_generation"] == "autoregressive"
    assert result["vision"]["model_id"] == "/local/blip"
    kwargs = {"local_files_only": True, "revision": "pinned"}
    assert events.count(("processor.load", "/local/blip", kwargs)) == 1
    assert events.count(("model.load", "/local/blip", kwargs)) == 1
    assert events.count(("model.eval",)) == 1
    assert events.count(("generate",)) == 2
    assert ("batch.to", "cpu") in events


def test_blip_rejects_non_rgb_before_loading(monkeypatch):
    def no_load(self):
        raise AssertionError("invalid BLIP input must not load a model")

    monkeypatch.setattr(BlipCaptioner, "_load", no_load)
    captioner = BlipCaptioner()
    with Image.new("L", (1, 1)) as image, pytest.raises(TypeError, match="RGB"):
        captioner(image)
    with pytest.raises(TypeError, match="RGB"):
        captioner("https://example.com/image.png")
