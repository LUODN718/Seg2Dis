"""Export an inference checkpoint without optimizer, RNG or local path metadata."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model, validate_checkpoint


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    validate_checkpoint(source)
    model = build_model(backbone_checkpoint='')
    model.load_state_dict(source['model_state'], strict=True)
    exported = {key: source[key] for key in ('model_state', 'epoch', 'variant', 'architecture',
                'residual_input_mode', 'backbone_name', 'image_size', 'type_weight', 'calibration_weight')}
    exported['source_checkpoint_sha256'] = digest(args.source)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    loaded = torch.load(destination, map_location='cpu', weights_only=True)
    validate_checkpoint(loaded)
    for key, value in source['model_state'].items():
        torch.testing.assert_close(value, loaded['model_state'][key], rtol=0, atol=0)
    manifest = {
        'model': 'complete SegDis-RGB', 'architecture': source['architecture'],
        'status': 'packaged_not_published', 'tag': None,
        'asset': destination.name, 'sha256': digest(destination),
        'source_checkpoint_sha256': exported['source_checkpoint_sha256'],
        'url': None,
        'selected_epoch': source['epoch'], 'backbone': source['backbone_name'],
        'residual_input': '[normalized original RGB, E, U]',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'contents': 'Exact model tensors and inference metadata; optimizer and RNG omitted',
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
