"""Optional curl range-download fallback for slow Hugging Face connections.

Uses the same pinned revisions as prepare.py. File sizes and digests come from
pinned Hub metadata; every artifact is checked before publishing the manifest.
The default endpoint is official; HF_ENDPOINT may select a public mirror.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
CHUNK = 4 * 1024 * 1024


def digest_file(path, lfs=True):
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, size, digest, lfs):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size and digest_file(path, lfs) == digest:
        return
    if size < CHUNK:
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '120', url,
                        '-o', str(path)], check=True)
    else:
        parts = path.parent / (path.name + '.parts')
        parts.mkdir(exist_ok=True)

        def part(index):
            start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
            target = parts / str(index)
            if target.exists() and target.stat().st_size == end - start + 1:
                return index
            temporary = parts / (str(index) + '.tmp')
            headers = parts / (str(index) + '.headers')
            subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '180',
                            '--range', f'{start}-{end}', '-D', str(headers),
                            url, '-o', str(temporary)], check=True)
            expected = f'content-range: bytes {start}-{end}/{size}'
            if expected not in headers.read_text().lower() or temporary.stat().st_size != end - start + 1:
                raise ValueError(f'Invalid HTTP range response: {path.name} part {index}')
            temporary.replace(target)
            headers.unlink()
            return index

        count = (size + CHUNK - 1) // CHUNK
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(part, index) for index in range(count)]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if done % 16 == 0 or done == count:
                    print(f'{path.name}: {done}/{count} parts', flush=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as output:
            for index in range(count):
                with (parts / str(index)).open('rb') as source:
                    shutil.copyfileobj(source, output)
        temporary.replace(path)
    if path.stat().st_size != size or digest_file(path, lfs) != digest:
        raise ValueError(f'File digest mismatch: {path}')
    if size >= CHUNK:
        shutil.rmtree(parts)


def main():
    from prepare import prepare_fixtures
    prepare_fixtures()
    definitions = [
        ('laya', 'convaiinnovations/laya', '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851',
         ['multilingual/encoder/config.json', 'multilingual/rl_agent_config.json',
          'multilingual/tokenizer/tokenizer.json', 'multilingual/tokenizer/tokenizer_config.json',
          'multilingual/model.safetensors']),
        ('blip', 'Salesforce/blip-image-captioning-base', '82a37760796d32b1411fe092ab5d4e227313294b',
         ['config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json',
          'special_tokens_map.json', 'vocab.txt', 'pytorch_model.bin']),
    ]
    manifest = {}
    for key, repo, revision, names in definitions:
        with urllib.request.urlopen(f'{ENDPOINT}/api/models/{repo}/revision/{revision}?blobs=true', timeout=60) as res:
            info = json.load(res)
        if info['sha'] != revision:
            raise ValueError('Model revision mismatch')
        files = {item['rfilename']: item for item in info['siblings']}
        folder = ROOT / 'downloads/vision-models' / key
        cache = ROOT / 'downloads/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        for name in names:
            item = files[name]
            lfs = 'lfs' in item
            digest = item['lfs']['sha256'] if lfs else item['blobId']
            destination = folder / name
            if (cache / name).is_file() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache / name, destination)
            print(f'Fetching/verifying {key}/{name}', flush=True)
            download(f'{ENDPOINT}/{repo}/resolve/{revision}/{name}', destination, item['size'], digest, lfs)
        weight = 'multilingual/model.safetensors' if key == 'laya' else 'pytorch_model.bin'
        manifest[key] = {'repo': repo, 'revision': revision, 'path': str(folder),
                         'subfolder': 'multilingual' if key == 'laya' else None,
                         'weight_sha256': files[weight]['lfs']['sha256'], 'download_endpoint': ENDPOINT}
    output = ROOT / 'artifacts/vision/models.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print('All files verified; local model manifest is ready.', flush=True)


if __name__ == '__main__':
    main()
