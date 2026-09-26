"""Download public weights and pinned scikit-image fixtures into project-local caches."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'downloads/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

FIXTURES = {
    'astronaut.png': '88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5',
    'camera.png': 'b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a',
    'chelsea.png': '596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb',
    'coffee.png': 'cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7',
    'horse.png': 'c7fb60789fe394c485f842291ea3b21e50d140f39d6dcb5fb9917cc178225455',
}

def prepare_fixtures():
    output = ROOT / 'artifacts/vision'
    image_dir = output / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, digest in FIXTURES.items():
        url = f'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/{name}'
        path = image_dir / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Fixture checksum mismatch: {name}')
        sources.append({'file': name, 'url': url, 'sha256': actual})
    (output / 'fixtures.json').write_text(json.dumps(sources, indent=2) + '\n')
    print('Verified fixtures', flush=True)
    return output


def main():
    output = prepare_fixtures()
    from huggingface_hub import snapshot_download
    models = [
        ('laya', 'convaiinnovations/laya', '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851',
         ['multilingual/rl_agent_config.json', 'multilingual/model.safetensors',
          'multilingual/tokenizer/*', 'multilingual/encoder/*']),
        ('blip', 'Salesforce/blip-image-captioning-base', '82a37760796d32b1411fe092ab5d4e227313294b',
         ['config.json', 'preprocessor_config.json', 'tokenizer*', 'special_tokens_map.json',
          'vocab.txt', 'pytorch_model.bin']),
    ]
    manifest = {}
    for key, repo, revision, patterns in models:
        print(f'Downloading {repo}@{revision}', flush=True)
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns, max_workers=3)
        weight_name = 'multilingual/model.safetensors' if key == 'laya' else 'pytorch_model.bin'
        expected = ('9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204' if key == 'laya'
                    else 'd6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767')
        digest = hashlib.sha256()
        with open(Path(path) / weight_name, 'rb') as weight:
            for chunk in iter(lambda: weight.read(1 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {repo}')
        manifest[key] = {'repo': repo, 'revision': revision, 'path': path,
                         'subfolder': 'multilingual' if key == 'laya' else None,
                         'weight_sha256': digest.hexdigest(),
                         'download_endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}
        (output / 'models.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
