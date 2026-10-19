import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def collect_predictions(model, loader, device, hybrid=False):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    elapsed = 0.0
    n_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            n_images += images.size(0)
            start = time.perf_counter()
            outputs = model(images)
            elapsed += time.perf_counter() - start

            if hybrid:
                probs = outputs
            else:
                probs = torch.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.concatenate(all_probs, axis=0),
        "seconds": elapsed,
        "images_per_second": n_images / elapsed if elapsed else 0.0,
        "ms_per_image": (elapsed / n_images) * 1000 if n_images else 0.0,
    }


def compute_metrics(labels, preds, class_names):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, preds, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "report": classification_report(
            labels,
            preds,
            target_names=class_names,
            zero_division=0,
            digits=4,
        ),
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(len(class_names)))
        ).tolist(),
    }


def summarize_split(name, outputs, class_names):
    scores = compute_metrics(outputs["labels"], outputs["preds"], class_names)
    scores["model"] = name
    scores["ms_per_image"] = outputs["ms_per_image"]
    scores["images_per_second"] = outputs["images_per_second"]
    return scores


def tune_hybrid_weight(cnn_probs, vit_probs, labels, class_names, steps=21):
    best = None

    for step in range(steps):
        cnn_weight = step / (steps - 1)
        mixed = cnn_weight * cnn_probs + (1.0 - cnn_weight) * vit_probs
        preds = mixed.argmax(axis=1)
        scores = compute_metrics(labels, preds, class_names)
        candidate = {
            "cnn_weight": cnn_weight,
            "vit_weight": 1.0 - cnn_weight,
            **{k: v for k, v in scores.items() if k != "report"},
        }
        if best is None or candidate["f1_macro"] > best["f1_macro"]:
            best = candidate
            best["report"] = scores["report"]

    return best


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def print_metrics(title, scores):
    print(title)
    print("  accuracy       : {:.4f}".format(scores["accuracy"]))
    print("  precision_macro: {:.4f}".format(scores["precision_macro"]))
    print("  recall_macro   : {:.4f}".format(scores["recall_macro"]))
    print("  f1_macro       : {:.4f}".format(scores["f1_macro"]))
    if "ms_per_image" in scores:
        print("  ms/image       : {:.2f}".format(scores["ms_per_image"]))
    if "cnn_weight" in scores:
        print(
            "  hybrid weights : CNN={:.2f}, ViT={:.2f}".format(
                scores["cnn_weight"], scores["vit_weight"]
            )
        )
