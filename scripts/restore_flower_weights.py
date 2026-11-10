"""Restore the GitHub-sized CLIP chunks without network access or Git LFS."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def restore():
    folder = ROOT / 'models/clip-vit-b32'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['sha256'] != EXPECTED:
        raise ValueError('Unexpected CLIP manifest hash')
    target = ROOT / 'train/.cache/weights/clip/ViT-B-32.pt'
    if target.is_file() and digest(target) == EXPECTED:
        print('CLIP SHA256 verified:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
            temporary = Path(out.name)
            for part in manifest['parts']:
                name = part['file']
                if Path(name).name != name:
                    raise ValueError('Invalid part path')
                source = folder / name
                if source.stat().st_size != part['size'] or digest(source) != part['sha256']:
                    raise ValueError(f'Invalid CLIP chunk: {name}')
                with source.open('rb') as f:
                    for block in iter(lambda: f.read(1024*1024), b''):
                        out.write(block)
        if temporary.stat().st_size != manifest['size'] or digest(temporary) != EXPECTED:
            raise ValueError('Restored CLIP hash mismatch')
        os.replace(temporary, target)
        print('CLIP restored and SHA256 verified:', target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    restore()
