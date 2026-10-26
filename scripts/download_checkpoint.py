"""Download the verified complete SegDis-RGB weights and check SHA256."""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/segdis_rgb_pvtb2_352.pth')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'benchmarks/rgb_release.json').read_text(encoding='utf-8'))
    if args.output.exists():
        if sha256(args.output) != manifest['sha256']:
            raise ValueError('Existing file has a different hash; choose a different --output')
        print('Checkpoint already verified:', args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.download')
    try:
        with urllib.request.urlopen(manifest['url']) as response, partial.open('wb') as handle:
            while block := response.read(1048576):
                handle.write(block)
        if sha256(partial) != manifest['sha256']:
            raise ValueError('Downloaded checkpoint failed SHA256 validation')
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)
    print('Verified RGB checkpoint:', args.output)


if __name__ == '__main__':
    main()
