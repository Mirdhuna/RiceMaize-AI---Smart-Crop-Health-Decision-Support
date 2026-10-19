import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import collect_predictions, compute_metrics, print_metrics, save_json
from models import HybridEnsemble, build_model
from torch_dataset import TorchCropDataset, ensure_resized_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Train RiceMaize AI models")
    parser.add_argument("--model", choices=["cnn", "vit", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vit-lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(batch_size, num_workers, seed):
    train_set = TorchCropDataset("train")
    val_set = TorchCropDataset("val")
    test_set = TorchCropDataset("test")
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def class_weights(dataset, device):
    counts = torch.zeros(len(dataset.class_names), dtype=torch.float)
    for row in dataset.rows:
        counts[int(row["class_id"])] += 1
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    return weights.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        if step % 10 == 0 or step == total_batches:
            batch_acc = correct / total
            avg_loss = running_loss / total
            print(
                f"  [{step:02d}/{total_batches:02d}] loss={avg_loss:.4f} acc={batch_acc:.4f}",
                flush=True,
            )

    return running_loss / total, correct / total


def evaluate_logits(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []
    elapsed = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            import time
            t0 = time.perf_counter()
            logits = model(images)
            elapsed += time.perf_counter() - t0
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    scores = compute_metrics(all_labels, all_preds, class_names)
    scores["loss"] = running_loss / total
    scores["ms_per_image"] = (elapsed / total) * 1000 if total else 0.0
    outputs = {"labels": all_labels, "preds": all_preds}
    return scores, outputs


def save_checkpoint(path, model, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        **extra,
    }
    torch.save(payload, path)


def train_model(name, args, device, loaders):
    train_set, val_set, _, train_loader, val_loader, _ = loaders
    class_names = train_set.class_names
    pretrained = (name == "vit") and (not args.no_pretrained)
    model = build_model(name, num_classes=len(class_names), pretrained=pretrained).to(device)

    lr = args.vit_lr if name == "vit" else args.lr
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set, device))

    best_f1 = -1.0
    best_path = args.checkpoints / "{}.pt".format(name)
    patience_left = args.patience
    history = []

    print("\nTraining {} on {} (pretrained={})".format(name.upper(), device, pretrained))
    print("Trainable parameters:", sum(p.numel() for p in params))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_scores, _ = evaluate_logits(model, val_loader, criterion, device, class_names)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_scores["loss"],
            "val_acc": val_scores["accuracy"],
            "val_f1_macro": val_scores["f1_macro"],
        }
        history.append(row)
        print(
            "Epoch {:02d}/{:02d}  train_loss={:.4f} acc={:.4f}  "
            "val_loss={:.4f} acc={:.4f} f1={:.4f}".format(
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_scores["loss"],
                val_scores["accuracy"],
                val_scores["f1_macro"],
            ),
            flush=True,
        )

        if val_scores["f1_macro"] > best_f1:
            best_f1 = val_scores["f1_macro"]
            patience_left = args.patience
            save_checkpoint(
                best_path,
                model,
                {
                    "name": name,
                    "class_names": class_names,
                    "val_metrics": {
                        key: value
                        for key, value in val_scores.items()
                        if key != "report"
                    },
                    "epoch": epoch,
                },
            )
            print("  saved best checkpoint to", best_path, flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping at epoch", epoch, flush=True)
                break

    save_json(args.results / "{}_history.json".format(name), {"history": history})
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, best_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    print("Checking / ensuring 224x224 image cache...")
    ensure_resized_cache()

    loaders = make_loaders(args.batch_size, args.num_workers, args.seed)
    names = ["cnn", "vit"] if args.model == "all" else [args.model]

    for name in names:
        train_model(name, args, device, loaders)

    print("\nTraining complete. Run evaluate.py to compare CNN, ViT, and hybrid.")


if __name__ == "__main__":
    main()
