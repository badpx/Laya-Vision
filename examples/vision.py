"""Run local image + text decisions using the checkpoints from research/vision/prepare.py."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / 'downloads/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--questions', type=Path, help='JSON mapping in Laya question format')
    parser.add_argument('--state', help='Optional additional text context')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--laya-model', type=Path, help='Local Laya checkpoint directory')
    parser.add_argument('--vision-model', type=Path, help='Local BLIP checkpoint directory')
    args = parser.parse_args()
    manifest_path = ROOT / 'artifacts/vision/models.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    laya_path = args.laya_model
    blip_path = args.vision_model
    if not laya_path and 'laya' in manifest:
        laya_path = Path(manifest['laya']['path']) / (manifest['laya']['subfolder'] or '')
    if not blip_path and 'blip' in manifest:
        blip_path = Path(manifest['blip']['path'])
    if not laya_path or not blip_path:
        parser.error('Run research/vision/prepare.py first, or supply both local model paths.')
    questions = json.loads(args.questions.read_text()) if args.questions else {
        'subject': {'type': 'choice', 'instructions': 'What is the main subject of the image?',
                    'criteria': {'person': 'a person', 'animal': 'an animal',
                                 'food': 'food or a drink', 'other': 'something else'}},
        'person_present': {'type': 'noul', 'instructions': 'Is a person visible in the image?'},
    }
    from laya import BlipCaptioner, VisionAgent, load
    agent = load(str(laya_path), device=args.device)
    captioner = BlipCaptioner(str(blip_path), device=args.device, local_files_only=True)
    result = VisionAgent(agent, captioner).predict(args.image, questions, state=args.state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
