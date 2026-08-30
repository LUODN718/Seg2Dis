import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config import CFG
from dataset import PairedCODDataset
from models import V3ForegroundTypeAllocationNet


def main():
    parser = argparse.ArgumentParser("V3 foreground type allocation inference")
    parser.add_argument("--path", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.device == "mps": device = torch.device("mps")
    elif args.device == "cpu": device = torch.device("cpu")
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()): device = torch.device(f"cuda:{args.gpu}")
    else: device = torch.device("cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = V3ForegroundTypeAllocationNet().to(device)
    raw_state = state.get("model_state", state)
    if any(key.startswith("v3.") for key in raw_state):
        model.load_state_dict(raw_state, strict=False)
    else:
        model.v3.load_compatible_model_state(raw_state)
    model.eval(); os.makedirs(args.output, exist_ok=True)
    loader = DataLoader(PairedCODDataset(args.path, CFG.image_size, False), batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            height, width = int(batch["original_size"][0][0]), int(batch["original_size"][1][0])
            stem = os.path.splitext(batch["name"][0])[0]
            for key in ("union", "sod", "cod"):
                probability = F.interpolate(output[key], (height, width), mode="bilinear", align_corners=False).sigmoid()[0, 0]
                array = (probability.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(os.path.join(args.output, f"{stem}_{key}.png"))


if __name__ == "__main__":
    main()
