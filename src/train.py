"""Train and evaluate the multimodal emotion model.

Synthetic data (no download needed):
    python src/train.py --data synthetic --epochs 30

Real AFFEC data (after scripts/download_affec.py + scripts/preprocess_affec.py):
    python src/train.py --data data/processed/affec_T50.npz --labels stimulus4 --split subject --folds 5

Splits
    random   trials shuffled across subjects (subject-dependent; same people in train and test)
    subject  whole subjects held out (subject-independent; the harder, more honest setting)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from data import LABEL_SCHEMES, MultimodalDataset, load_affec, make_synthetic
from model import MODALITIES, MultimodalEmotionNet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, preds, labels, probs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for eeg, phy, eye, mask, y in loader:
            eeg, phy, eye, mask, y = (t.to(device) for t in (eeg, phy, eye, mask, y))
            logits = model(eeg, phy, eye, mask)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            probs.append(torch.softmax(logits, 1).detach().cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return dict(loss=loss_sum / len(labels), acc=accuracy_score(labels, preds),
                f1=f1_score(labels, preds, average="macro", zero_division=0),
                preds=preds, labels=labels, probs=torch.cat(probs).numpy())


def make_splits(data, args):
    """Yield (train_idx, val_idx, test_idx) for each fold."""
    y, groups, idx = data["labels"], data["groups"], np.arange(len(data["labels"]))
    rng = np.random.RandomState(args.seed)
    if args.split == "subject":
        if args.folds > 1:
            outer = GroupKFold(n_splits=args.folds).split(idx, y, groups)
        else:
            outer = GroupShuffleSplit(1, test_size=args.test_size, random_state=args.seed).split(idx, y, groups)
        for trv, te in outer:
            inner = GroupShuffleSplit(1, test_size=args.val_size, random_state=args.seed)
            tr, va = next(inner.split(trv, y[trv], groups[trv]))
            yield trv[tr], trv[va], te
    else:
        if args.folds > 1:
            outer = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(idx, y)
        else:
            outer = [train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)]
        for trv, te in outer:
            tr, va = train_test_split(trv, test_size=args.val_size, stratify=y[trv],
                                      random_state=rng.randint(1 << 30))
            yield tr, va, te


def train_fold(data, tr, va, te, args, device, out_dir, fold):
    set_seed(args.seed + fold)
    n_classes = len(data["classes"])
    dl = lambda i, shuffle, p=0.0: DataLoader(MultimodalDataset(data, i, p), batch_size=args.batch_size,  # noqa
                                              shuffle=shuffle, num_workers=0)
    train_dl, val_dl, test_dl = dl(tr, True, args.modality_dropout), dl(va, False), dl(te, False)

    model = MultimodalEmotionNet(n_classes, args.modalities, data["physio"].shape[-1], d_model=args.d_model,
                                 n_heads=args.heads, n_layers=args.layers, dropout=args.dropout).to(device)
    counts = np.bincount(data["labels"][tr], minlength=n_classes)
    weights = torch.tensor(len(tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = os.path.join(out_dir, f"model_fold{fold}.pt")
    best, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, train_dl, criterion, device, optimizer)
        vam = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=trm["loss"], train_acc=trm["acc"],
                            val_loss=vam["loss"], val_acc=vam["acc"], val_f1=vam["f1"]))
        print(f"[fold {fold}] epoch {epoch:02d} | train loss {trm['loss']:.3f} acc {trm['acc']:.3f} | "
              f"val loss {vam['loss']:.3f} acc {vam['acc']:.3f} f1 {vam['f1']:.3f} | {time.time() - t0:.0f}s",
              flush=True)
        if vam["f1"] > best:
            best, bad = vam["f1"], 0
            torch.save({"state_dict": model.state_dict(), "classes": data["classes"],
                        "config": vars(args)}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[fold {fold}] early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    tem = run_epoch(model, test_dl, criterion, device)
    tem["balanced_acc"] = balanced_accuracy_score(tem["labels"], tem["preds"])
    return tem, history


def save_plots(history, cm, classes, out_dir, fold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(title="Cross-entropy loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val")
    ax[1].axhline(1 / len(classes), color="grey", ls=":", label="chance")
    ax[1].set(title="Accuracy", xlabel="epoch")
    for a in ax[:2]:
        a.legend()
    im = ax[2].imshow(cm, cmap="Blues")
    ax[2].set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
              yticklabels=classes, xlabel="predicted", ylabel="true", title="Test confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax[2].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"results_fold{fold}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic", help="'synthetic' or path to processed .npz")
    ap.add_argument("--labels", default="stimulus4", choices=list(LABEL_SCHEMES))
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    ap.add_argument("--split", default="subject", choices=["subject", "random"])
    ap.add_argument("--folds", type=int, default=1, help=">1 for k-fold cross-validation")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--samples", type=int, default=2000, help="synthetic only")
    ap.add_argument("--noise", type=float, default=3.0, help="synthetic only: noise level")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--modality-dropout", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory (default outputs/<auto-name>)")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data == "synthetic":
        data = make_synthetic(n=args.samples, noise=args.noise, seed=args.seed)
        tag = "synthetic"
    else:
        data = load_affec(args.data, args.labels)
        tag = f"affec_{args.labels}"
    name = f"{tag}_{args.split}_{'+'.join(args.modalities)}"
    out_dir = args.out or os.path.join("outputs", name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(data['labels'])} samples, {len(set(data['groups']))} subjects, classes {data['classes']}, "
          f"class counts {np.bincount(data['labels']).tolist()}, device {device}")

    results, all_preds, all_labels = [], [], []
    for fold, (tr, va, te) in enumerate(make_splits(data, args)):
        print(f"\n=== fold {fold}: train {len(tr)}  val {len(va)}  test {len(te)} ===")
        tem, history = train_fold(data, tr, va, te, args, device, out_dir, fold)
        cm = confusion_matrix(tem["labels"], tem["preds"], labels=range(len(data["classes"])))
        save_plots(history, cm, data["classes"], out_dir, fold)
        print(f"[fold {fold}] TEST acc {tem['acc']:.3f}  macro-F1 {tem['f1']:.3f}  "
              f"balanced acc {tem['balanced_acc']:.3f}")
        results.append({k: float(tem[k]) for k in ("acc", "f1", "balanced_acc")})
        all_preds.append(tem["preds"])
        all_labels.append(tem["labels"])

    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)
    report = classification_report(labels, preds, labels=range(len(data["classes"])),
                                   target_names=data["classes"], zero_division=0, digits=3)
    summary = {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results])))
               for k in results[0]}
    summary["chance"] = 1 / len(data["classes"])
    print("\n" + report)
    print("summary (mean, std over folds):", json.dumps(summary))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(dict(config=vars(args), folds=results, summary=summary, n_samples=int(len(labels)),
                       confusion_matrix=confusion_matrix(labels, preds).tolist(), report=report), f, indent=2)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
