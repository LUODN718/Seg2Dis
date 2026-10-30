"""Check source cleanup against a Git revision with the released RGB weights."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_usc12k import USC12KDataset
from models import build_model, validate_checkpoint

KEYS = ("master_final", "base_final", "final", "union", "semantic_logits",
        "type_probability", "residual_input", "three_class_probability")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_commit", required=True)
    parser.add_argument("--usc_root", required=True)
    parser.add_argument("--saved_predictions", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with Path(args.checkpoint).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_hash = digest.hexdigest()
    manifest = json.loads((ROOT / "benchmarks/rgb_release.json").read_text(encoding="utf-8-sig"))
    if checkpoint_hash not in {manifest["sha256"], manifest["source_checkpoint_sha256"]}:
        raise ValueError("Checkpoint hash does not match the released RGB model")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(state)
    torch.set_num_threads(4)
    model = build_model(backbone_checkpoint="").eval()
    model.load_state_dict(state["model_state"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == manifest["parameter_count"]
    assert not any("haar" in type(module).__name__.lower() for module in model.modules())
    cases = []
    for scene in ("Scene-A", "Scene-B", "Scene-C", "Scene-D"):
        item = USC12KDataset(args.usc_root, scene, 352, training=False)[0]
        cases.append((scene, item["sample_id"], item["image"]))

    reference_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.reference_commit + "^{commit}"],
        cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", reference_commit], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="segdis_source_cleanup_") as temporary:
        temporary = Path(temporary)
        reference_root = temporary / "reference"
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            source.extractall(reference_root)
        inputs_path = temporary / "inputs.pt"
        outputs_path = temporary / "outputs.pt"
        torch.save([case[2] for case in cases], inputs_path)
        baseline_code = '''
import sys, torch
sys.path.insert(0, sys.argv[1])
from models import build_model, validate_checkpoint
torch.set_num_threads(4)
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
validate_checkpoint(state)
model = build_model(backbone_checkpoint="").eval()
model.load_state_dict(state["model_state"], strict=True)
inputs = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
keys = sys.argv[5:]
with torch.inference_mode():
    outputs = [{key: result[key] for key in keys}
               for image in inputs for result in [model(image[None])]]
torch.save({"outputs": outputs,
            "state_shapes": {key: list(value.shape) for key, value in model.state_dict().items()}}, sys.argv[4])
'''
        subprocess.run([sys.executable, "-c", baseline_code, str(reference_root),
                        str(Path(args.checkpoint).resolve()), str(inputs_path),
                        str(outputs_path), *KEYS], cwd=reference_root, check=True)
        reference = torch.load(outputs_path, map_location="cpu", weights_only=True)
        assert reference["state_shapes"] == {
            key: list(value.shape) for key, value in model.state_dict().items()}
        projection_observations = []

        def observe(name):
            def hook(module, inputs, output):
                count = int(torch.count_nonzero(inputs[0]))
                assert count == 0, name
                projection_observations.append({"name": name, "input_nonzero_count": count,
                                               "output_max_abs": float(output.abs().max())})
            return hook

        projections = [(name, module) for name, module in model.named_modules()
                       if name.endswith(("frequency_projection", "smooth_frequency", "sharp_frequency"))]
        assert len(projections) == 5
        handles = [module.register_forward_hook(observe(name)) for name, module in projections]
        results = []
        try:
            with torch.inference_mode():
                for (scene, sample_id, image), expected in zip(cases, reference["outputs"]):
                    output = model(image[None])
                    errors = {}
                    for key in KEYS:
                        torch.testing.assert_close(output[key], expected[key], rtol=0, atol=0)
                        errors[key] = float((output[key] - expected[key]).abs().max())
                    labels = output["three_class_probability"].argmax(1)[0].numpy()
                    with Image.open(Path(args.saved_predictions) / scene / f"{sample_id}.png") as prediction:
                        rgb = np.array(prediction.convert("RGB"))
                    stored = (rgb[:, :, 0] == 255).astype(np.uint8) + 2 * (rgb[:, :, 1] == 255).astype(np.uint8)
                    assert np.array_equal(labels, stored), sample_id
                    results.append({"scene": scene, "sample_id": sample_id,
                                    "max_absolute_error": errors, "matches_saved_prediction": True,
                                    "compatibility_projections": projection_observations[-5:]})
                    print(scene, sample_id, "PASS", flush=True)
        finally:
            for handle in handles:
                handle.remove()
    report = {"model": "complete SegDis-RGB", "source_checkpoint_sha256": checkpoint_hash,
              "reference_commit": reference_commit, "selected_epoch": state["epoch"],
              "strict_state_load": True, "state_keys_and_shapes_unchanged": True,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "haar_module_count": 0, "device": "CPU", "image_size": 352,
              "torch_version": str(torch.__version__), "cases": results,
              "scope": "One real image per scene; not a full-dataset re-evaluation",
              "fresh_training_note": "Removing discarded module initialization changes RNG consumption"}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
