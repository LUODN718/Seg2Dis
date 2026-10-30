"""Download the verified complete SegDis-RGB weights and check SHA256."""
import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from urllib.parse import urlparse
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
    parser.add_argument('--use_git_credentials', action='store_true',
                        help='Use the configured Git credential helper for this private repository')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'benchmarks/rgb_release.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'trained_release' or not manifest.get('sha256') or not manifest.get('url'):
        raise RuntimeError(
            'No trained segdis_rgb_v2 checkpoint has been published. Train the current '
            'architecture first; the historical RGB release belongs to the previous structure.'
        )
    if args.output.exists():
        if sha256(args.output) != manifest['sha256']:
            raise ValueError('Existing file has a different hash; choose a different --output')
        print('Checkpoint already verified:', args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.download')
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if args.use_git_credentials and not token:
        result = subprocess.run(['git', 'credential', 'fill'],
            input=chr(10).join(['protocol=https', 'host=github.com', '', '']),
            capture_output=True, text=True, check=True)
        credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        token = credential.get('password')
    headers = {'User-Agent': 'SegDis-RGB-downloader'}
    url = manifest['url']
    if token:
        headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/octet-stream'})
        url = manifest['api_asset_url']

    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, response_headers, newurl):
            redirected = super().redirect_request(request, fp, code, msg, response_headers, newurl)
            if redirected and urlparse(newurl).hostname != urlparse(request.full_url).hostname:
                redirected.remove_header('Authorization')
            return redirected

    opener = urllib.request.build_opener(SafeRedirect())
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=300) as response, partial.open('wb') as handle:
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
