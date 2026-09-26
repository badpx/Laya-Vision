"""Probe caption information loss and image-plus-text behavior using real Laya weights.

Reuses captions from evaluate.py; these probes measure the bridge's text bottleneck,
not a new vision inference. Oracle descriptions isolate the missing visual detail.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

PROBES = [
    ('astronaut.png', 'What color is the astronaut suit?',
     {'orange': 'orange', 'white': 'white', 'blue': 'blue'}, 'orange',
     'A woman wearing an orange astronaut suit holds a black helmet.'),
    ('camera.png', 'What supports the camera?',
     {'tripod': 'a tripod', 'hands': 'held only by hands', 'table': 'a table'}, 'tripod',
     'A man looks through a camera mounted on a tripod.'),
    ('chelsea.png', "What color are the cat's eyes?",
     {'green': 'green', 'blue': 'blue', 'red': 'red'}, 'green',
     'A tabby cat with green eyes.'),
    ('coffee.png', 'What color is the outside of the coffee cup?',
     {'red': 'red or reddish brown', 'white': 'white', 'blue': 'blue'}, 'red',
     'A cup of coffee. The outside of the cup is red and the inside is white.'),
    ('horse.png', 'Which way is the horse facing?',
     {'left': 'toward the left side of the image', 'right': 'toward the right side of the image'}, 'right',
     'A black silhouette of a horse facing the right side of the image.'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=ROOT / 'artifacts/vision/evaluation-mps-en.json')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    import torch
    from laya import load
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'artifacts/vision/models.json').read_text())
    path = Path(manifest['laya']['path']) / manifest['laya']['subfolder']
    agent = load(str(path), device=args.device)
    evaluation = json.loads(args.evaluation.read_text())
    descriptions = {row['image']: row['result']['vision']['state'] for row in evaluation['cases']}
    rows = []
    for image, instruction, criteria, expected, oracle in PROBES:
        questions = {'detail': {'type': 'choice', 'instructions': instruction, 'criteria': criteria}}
        bridge = agent.predict(descriptions[image], questions)
        oracle_result = agent.predict('Image description: ' + oracle, questions)
        rows.append({'image': image, 'state': descriptions[image], 'questions': questions,
                     'expected': expected, 'bridge': bridge, 'oracle': oracle_result,
                     'bridge_correct': bridge['answers']['detail']['choice'] == expected,
                     'oracle_correct': oracle_result['answers']['detail']['choice'] == expected})
    mixed = []
    for policy, expected in [('Images of animals belong in queue alpha. All other images belong in queue beta.', 'alpha'),
                             ('Images of animals belong in queue beta. All other images belong in queue alpha.', 'beta')]:
        state = descriptions['chelsea.png'] + '\n\nAdditional context:\n' + policy
        questions = {'queue': {'type': 'choice', 'instructions': 'Which queue should this image be sent to under the stated policy?',
                               'criteria': {'alpha': 'queue alpha', 'beta': 'queue beta'}}}
        result = agent.predict(state, questions)
        mixed.append({'state': state, 'expected': expected, 'result': result,
                      'correct': result['answers']['queue']['choice'] == expected})
    report = {'scope': '5 detail probes and 2 context-switch probes; captions reused from main evaluation.',
              'device': str(agent.device), 'detail_correct': sum(row['bridge_correct'] for row in rows),
              'oracle_correct': sum(row['oracle_correct'] for row in rows),
              'mixed_context_correct': sum(row['correct'] for row in mixed), 'cases': rows, 'mixed_context': mixed}
    output = ROOT / 'artifacts/vision/probes.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
