import argparse
import json
from pathlib import Path

import torch

from dataset import load_class_names
from models import HybridEnsemble, build_model
from torch_dataset import load_image_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a rice or maize image")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        choices=["cnn", "vit", "hybrid"],
        default="hybrid",
        help="Which trained system to use",
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_model(name, class_names, checkpoint_dir, device):
    if name == "hybrid":
        cnn = load_model("cnn", class_names, checkpoint_dir, device)
        vit = load_model("vit", class_names, checkpoint_dir, device)
        weights_path = checkpoint_dir / "hybrid.json"
        cnn_weight = 0.5
        if weights_path.exists():
            cnn_weight = json.loads(weights_path.read_text(encoding="utf-8"))["cnn_weight"]
        model = HybridEnsemble(cnn, vit, cnn_weight=cnn_weight)
        model.to(device)
        model.eval()
        return model

    path = checkpoint_dir / "{}.pt".format(name)
    if not path.exists():
        raise FileNotFoundError("Missing {}. Train the models first.".format(path))

    model = build_model(name, num_classes=len(class_names), pretrained=False)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    if not args.image.exists():
        raise SystemExit("Image not found: {}".format(args.image))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()
    model = load_model(args.model, class_names, args.checkpoints, device)
    tensor = load_image_tensor(args.image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        if args.model == "hybrid":
            probs = outputs
        else:
            probs = torch.softmax(outputs, dim=1)

    probs = probs.squeeze(0).cpu()
    confidence, predicted = probs.max(dim=0)
    topk = torch.topk(probs, k=min(args.top_k, len(class_names)))

    print("Model:", args.model)
    print("Predicted class:", class_names[int(predicted)])
    print("Confidence: {:.2f}%".format(float(confidence) * 100))
    print("Top predictions:")
    for score, index in zip(topk.values, topk.indices):
        print("  {:6.2f}%  {}".format(float(score) * 100, class_names[int(index)]))


if __name__ == "__main__":
    main()
