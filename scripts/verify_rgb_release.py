"""Verify a complete RGB checkpoint against its original implementation on CPU."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_usc12k import USC12KDataset
from models import build_model, validate_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference_snapshot', required=True)
    parser.add_argument('--usc_root', required=True)
    parser.add_argument('--saved_predictions', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(state)
    model = build_model(backbone_checkpoint='').eval()
    model.load_state_dict(state['model_state'], strict=True)
    cases = []
    for scene in ('Scene-A', 'Scene-B', 'Scene-C', 'Scene-D'):
        dataset = USC12KDataset(args.usc_root, scene, 352, training=False)
        item = dataset[0]
        cases.append((scene, item['sample_id'], item['image']))
    keys = ('three_class_probability', 'union', 'semantic_logits', 'residual_input')
    with tempfile.TemporaryDirectory(prefix='segdis_rgb_release_verify_') as temp:
        temp = Path(temp)
        torch.save([item[2] for item in cases], temp / 'inputs.pt')
        baseline = '''
import sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from models.rgb_input_ablation import build_input_ablation_model
torch.set_num_threads(4)
state = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
model = build_input_ablation_model('rgb', backbone_checkpoint='').eval()
model.load_state_dict(state['model_state'], strict=True)
inputs = torch.load(sys.argv[3], map_location='cpu', weights_only=True)
results = []
with torch.inference_mode():
    for image in inputs:
        output = model(image[None])
        results.append({k: output[k] for k in ('three_class_probability','union','semantic_logits','residual_input')})
torch.save(results, sys.argv[4])
'''
        subprocess.run([sys.executable, '-c', baseline, str(Path(args.reference_snapshot).resolve()),
                        args.checkpoint, str(temp / 'inputs.pt'), str(temp / 'outputs.pt')],
                       cwd=args.reference_snapshot, check=True)
        reference = torch.load(temp / 'outputs.pt', map_location='cpu', weights_only=True)
        results = []
        with torch.inference_mode():
            for (scene, sid, image), expected in zip(cases, reference):
                output = model(image[None])
                errors = {}
                for key in keys:
                    torch.testing.assert_close(output[key], expected[key], rtol=0, atol=1e-6)
                    errors[key] = float((output[key] - expected[key]).abs().max())
                torch.testing.assert_close(output['residual_input'][:, :3], image[None], rtol=0, atol=0)
                labels = output['three_class_probability'].argmax(1)[0].numpy()
                with Image.open(Path(args.saved_predictions) / scene / f'{sid}.png') as im:
                    rgb = np.array(im.convert('RGB'))
                stored = (rgb[:, :, 0] == 255).astype(np.uint8) + 2 * (rgb[:, :, 1] == 255).astype(np.uint8)
                assert np.array_equal(labels, stored), f'Saved prediction mismatch: {sid}'
                results.append({'scene': scene, 'sample_id': sid, 'max_absolute_error': errors,
                                'matches_saved_prediction': True})
                print(scene, sid, 'PASS', flush=True)
    report = {'model': 'complete SegDis-RGB', 'source_checkpoint_sha256': sha256(args.checkpoint),
              'selected_epoch': state['epoch'], 'strict_state_load': True, 'device': 'CPU',
              'image_size': 352, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'reference': 'Original RGB experiment source snapshot', 'cases': results,
              'scope': 'One real case from each scene; not a fresh full-dataset benchmark'}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
