# Local image decisions

`VisionAgent` adds a local image-captioning step before an existing Laya `Agent` or `Router`.
BLIP describes the image, then Laya answers your typed questions using that description and
optional text context. This is a caption bridge: the Laya checkpoint remains a text model,
and BLIP generates its description autoregressively.

## Install

From this source checkout:

```bash
python -m pip install -e ".[vision]"
```

The `vision` extra adds the image dependencies. Prepare complete local checkpoints in
`./models/laya` and `./models/blip` before running the example. The Laya directory must contain
its weights, `rl_agent_config.json`, `encoder/`, and `tokenizer/`; the BLIP directory must
contain the model and processor files for
[`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base).

## Predict from a path or PIL image

```python
import laya
from laya import BlipCaptioner, VisionAgent

agent = laya.load("./models/laya", device="cpu")
captioner = BlipCaptioner(
    "./models/blip",
    device="cpu",
    local_files_only=True,
    max_new_tokens=48,
)
vision = VisionAgent(agent, captioner=captioner)

questions = {
    "subject": {
        "type": "choice",
        "instructions": "What is the main subject described in the image?",
        "criteria": {
            "animal": "an animal",
            "vehicle": "a vehicle",
            "other": "another subject",
        },
    },
}
result = vision.predict(
    "./photo.jpg",
    questions,
    state={"task": "Categorize the main subject of this photograph."},
)
print(result["answers"]["subject"]["choice"])
print(result["vision"]["caption"])
```

Pass a PIL image in the same position:

```python
from PIL import Image

with Image.open("./photo.jpg") as image:
    result = vision.predict(image, questions)
```

Each call accepts one local file path or one PIL image, with at most 20 million pixels.
EXIF orientation is applied and the input is converted to RGB. Image URLs, animated images,
multi-page files, and lists of images are rejected. The caller retains ownership of a supplied
PIL image.

## Loading and offline operation

`BlipCaptioner` loads its optional libraries, processor, and weights on its first call, then
reuses them. Constructing it does not load BLIP. The wrapped Laya agent follows its own loading
behavior; for example, `laya.load(...)` loads the text checkpoint immediately.

Use a local BLIP directory with `local_files_only=True` to prevent model downloads. The default
`BlipCaptioner()` uses the Hugging Face model ID and may download weights on first use. Image
processing and inference run on the local device. If `device` is omitted, BLIP selects CUDA,
then MPS, then CPU. Set each model's device explicitly when you need a particular placement.

An existing `Router` also works as the wrapped agent. Extra keyword arguments to
`vision.predict(...)`, such as `max_len`, `head_max_len`, hooks, or a Router's `model` override,
are forwarded to its `predict(...)` method. A custom `captioner` can be any callable that
accepts one RGB PIL image and returns a nonempty description string.

## Read the result and its limits

The result preserves Laya's answers and adds a `vision` dictionary:

| Field | Meaning |
|---|---|
| `vision.caption` | The description supplied by the captioner |
| `vision.state` | The combined text submitted to Laya, before its hooks and tokenization |
| `vision.mode` | `caption_bridge` |
| `vision.probability_scope` | `text_surrogate_only`: probabilities describe the text decision |
| `vision.usage_scope` | `laya_text_stage_only` |
| `vision.end_to_end_single_pass` | `False`: captioning and Laya inference are separate stages |
| `vision.image` | Width, height, and RGB mode after orientation handling |
| `vision.timings_ms` | Image preparation, captioning, Laya, and total elapsed times |

Laya's probabilities do not account for errors or omissions in the image description, and
are not calibrated probabilities about the original pixels. Inspect the caption when evaluating
an answer. Small text, exact counts, and subtle spatial relationships can be missing or wrong;
validate the whole pipeline on representative images before choosing confidence thresholds.

The returned `usage` counts only the Laya text stage. An `output_tokens` value of zero does
not include BLIP's generated caption tokens. On the first call, the caption timing also includes
lazy loading, so compare warm calls separately from startup.

The combined state starts with `Image description: ...`, followed by optional `Additional
context:`. A dict or list context is serialized as JSON. This combined input is a string and
is subject to Laya's usual token budget and truncation: the metadata records the submitted
text, not a guarantee that every token reached the model. Keep context short or set a supported
`max_len` and `head_max_len` for the wrapped checkpoint. Long context may lose information,
including recent turns in a serialized conversation list.
