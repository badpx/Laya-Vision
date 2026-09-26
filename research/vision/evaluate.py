"""Real-checkpoint smoke evaluation, with oracle-text and mismatched-caption controls.

This small public-fixture suite tests integration, not generalization or calibration.
Weights and images are fetched separately; this script runs offline by default.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / 'downloads/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

CASES = [
    {'image': 'astronaut.png', 'category': 'person', 'person': True, 'animal': False, 'people': 1,
     'oracle': 'A photograph of one woman wearing an orange astronaut space suit.'},
    {'image': 'camera.png', 'category': 'person', 'person': True, 'animal': False, 'people': 1,
     'oracle': 'A black and white photograph of one man standing behind a camera on a tripod.'},
    {'image': 'chelsea.png', 'category': 'animal', 'person': False, 'animal': True, 'people': 0,
     'oracle': 'A close-up photograph of a tabby cat lying down.'},
    {'image': 'coffee.png', 'category': 'food', 'person': False, 'animal': False, 'people': 0,
     'oracle': 'A cup of coffee on a saucer on a table.'},
    {'image': 'horse.png', 'category': 'animal', 'person': False, 'animal': True, 'people': 0,
     'oracle': 'A black silhouette of a horse on a white background.'},
]
QUESTIONS = {
    'category': {'type': 'choice', 'instructions': 'What is the main subject of the image?',
                 'criteria': {'person': 'a person', 'animal': 'an animal',
                              'food': 'food or a drink', 'other': 'something else'}},
    'person': {'type': 'noul', 'instructions': 'Is a person visible in the image?'},
    'animal': {'type': 'noul', 'instructions': 'Is an animal visible in the image?'},
    'people': {'type': 'score', 'instructions': 'How many people are visible in the image?',
               'criteria': ['no people', 'one person', 'two or more people']},
}

QUESTIONS_ZH = {
    'category': {'type': 'choice', 'instructions': '图像中的主要主体是什么？',
                 'criteria': {'person': '人物', 'animal': '动物', 'food': '食物或饮料', 'other': '其他东西'}},
    'person': {'type': 'noul', 'instructions': '图像中有人吗？'},
    'animal': {'type': 'noul', 'instructions': '图像中有动物吗？'},
    'people': {'type': 'score', 'instructions': '图像中有多少人？', 'criteria': ['没有人', '一个人', '两个人或更多人']},
}


def validate_answers(answers):
    if set(answers) != set(QUESTIONS):
        raise AssertionError('Missing or unexpected typed answers')
    for key, answer in answers.items():
        if answer['type'] != QUESTIONS[key]['type']:
            raise AssertionError('Incorrect answer type')
        if answer['type'] == 'noul':
            values = [answer['noul']]
        else:
            values = list(answer['probabilities'].values())
            if abs(sum(values) - 1) > 0.001:
                raise AssertionError('Distribution does not sum to one')
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise AssertionError('Non-finite or out-of-range probability')
        if answer['type'] == 'score' and not 0 <= answer['score'] <= len(values) - 1:
            raise AssertionError('Ordinal score outside its range')


def score(answers, case):
    validate_answers(answers)
    labels = {key: case[key] for key in QUESTIONS}
    predicted = {
        'category': answers['category']['choice'],
        'person': answers['person']['noul'] >= 0.5,
        'animal': answers['animal']['noul'] >= 0.5,
        'people': int(max(answers['people']['probabilities'],
                          key=answers['people']['probabilities'].get)),
    }
    return {'expected': labels, 'predicted': predicted,
            'correct': {key: predicted[key] == labels[key] for key in QUESTIONS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/vision/evaluation.json')
    args = parser.parse_args()
    questions = QUESTIONS_ZH if args.language == 'zh' else QUESTIONS
    import torch
    from laya import BlipCaptioner, VisionAgent, load
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    manifest = json.loads((ROOT / 'artifacts/vision/models.json').read_text())
    laya_path = Path(manifest['laya']['path']) / (manifest['laya']['subfolder'] or '')
    blip_path = manifest['blip']['path']
    start = time.perf_counter()
    agent = load(str(laya_path), device=args.device)
    captioner = BlipCaptioner(blip_path, device=args.device, local_files_only=True)
    vision = VisionAgent(agent, captioner)
    load_ms = (time.perf_counter() - start) * 1000
    rows = []
    # First image is repeated as warmup, excluded from measured rows.
    warmup = vision.predict(ROOT / 'artifacts/vision/images' / CASES[0]['image'], questions)
    for case in CASES:
        result = vision.predict(ROOT / 'artifacts/vision/images' / case['image'], questions)
        rows.append({'image': case['image'], 'result': result,
                     'vision_score': score(result['answers'], case)})
        print(json.dumps({'image': case['image'], 'caption': result['vision']['caption'],
                          'score': rows[-1]['vision_score']}, ensure_ascii=False), flush=True)
    for index, (case, row) in enumerate(zip(CASES, rows)):
        # Exact same wrapper state for controls; only the sensory description changes.
        from PIL import Image
        oracle = VisionAgent(agent, lambda image, text=case['oracle']: text)
        shifted = rows[(index + 2) % len(rows)]['result']['vision']['caption']
        mismatch = VisionAgent(agent, lambda image, text=shifted: text)
        blank = VisionAgent(agent, lambda image: 'No visual description is available.')
        with Image.open(ROOT / 'artifacts/vision/images' / case['image']) as image:
            for name, bridge in [('oracle', oracle), ('mismatch', mismatch), ('no_visual', blank)]:
                answer = bridge.predict(image, questions)
                row[name + '_result'] = answer
                row[name + '_score'] = score(answer['answers'], case)
    summary = {}
    for variant in ['vision', 'oracle', 'mismatch', 'no_visual']:
        correct = sum(sum(row[variant + '_score']['correct'].values()) for row in rows)
        summary[variant] = {'correct': correct, 'total': len(rows) * len(QUESTIONS),
                            'accuracy': correct / (len(rows) * len(QUESTIONS)),
                            'by_question': {q: sum(r[variant + '_score']['correct'][q] for r in rows)
                                            / len(rows) for q in QUESTIONS}}
    timing_keys = rows[0]['result']['vision']['timings_ms'].keys()
    timings = {key: {'median': statistics.median(r['result']['vision']['timings_ms'][key] for r in rows),
                     'min': min(r['result']['vision']['timings_ms'][key] for r in rows),
                     'max': max(r['result']['vision']['timings_ms'][key] for r in rows)}
               for key in timing_keys}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Integration smoke test on 5 public fixtures / 20 labels, not a held-out benchmark.',
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                           'device': str(agent.device), 'requested_device': args.device,
                           'threads': args.threads, 'mps_available': torch.backends.mps.is_available()},
              'versions': {name: importlib.metadata.version(name)
                           for name in ['torch', 'torchvision', 'transformers', 'huggingface_hub', 'pillow', 'laya']},
              'models': {key: {k: v for k, v in value.items() if k != 'path'}
                         for key, value in manifest.items()},
              'language': args.language, 'questions': questions, 'laya_load_ms': load_ms, 'warmup_includes_blip_load': True, 'warmup': warmup['vision'],
              'summary': summary, 'timings_ms': timings, 'cases': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'timings_ms': timings, 'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
