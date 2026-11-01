"""Download only a verified release identified by the current manifest."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.runtime import ROOT,sha256

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'benchmarks/release.json')
    p.add_argument('--output',type=Path,default=ROOT/'checkpoints/segdis.pth')
    args=p.parse_args();manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('status')!='verified_release' or not manifest.get('url') or not manifest.get('sha256'):
        raise RuntimeError('Matching trained and verified weights are not published yet')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.download')
    if temp.exists():raise FileExistsError(temp)
    try:
        urllib.request.urlretrieve(manifest['url'],temp)
        if sha256(temp)!=manifest['sha256']:raise ValueError('Downloaded checkpoint SHA256 mismatch')
        temp.replace(args.output)
    finally:
        if temp.exists():temp.unlink()

if __name__=='__main__':main()
