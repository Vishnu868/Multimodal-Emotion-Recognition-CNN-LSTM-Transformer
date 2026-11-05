"""Train and evaluate the multimodal emotion model."""
import argparse

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split

from data import EMOTIONS, MultimodalDataset, make_synthetic
from model import MultimodalEmotionNet


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum, preds, labels = 0, 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for eeg, phy, eye, y in loader:
            eeg, phy, eye, y = eeg.to(device), phy.to(device), eye.to(device), y.to(device)
            logits = model(eeg, phy, eye)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            p = logits.argmax(1)
            loss_sum += loss.item() * len(y)
            correct += (p == y).sum().item()
            total += len(y)
            preds += p.cpu().tolist()
            labels += y.cpu().tolist()
    return loss_sum / total, correct / total, preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eeg, physio, eye, y = make_synthetic(n=args.samples)
    ds = MultimodalDataset(eeg, physio, eye, y)
    n_val = n_test = int(0.15 * len(ds))
    train_ds, val_ds, test_ds = random_split(ds, [len(ds) - n_val - n_test, n_val, n_test])
    dl = lambda d, s: DataLoader(d, batch_size=args.batch_size, shuffle=s)  # noqa: E731

    model = MultimodalEmotionNet(n_classes=len(EMOTIONS), physio_features=physio.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta, *_ = run_epoch(model, dl(train_ds, True), criterion, device, optimizer)
        vl, va, *_ = run_epoch(model, dl(val_ds, False), criterion, device)
        print(f"epoch {epoch:02d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
        if va >= best:
            best = va
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    _, acc, preds, labels = run_epoch(model, dl(test_ds, False), criterion, device)
    print(f"\nTest accuracy: {acc:.3f}")
    print(classification_report(labels, preds, labels=list(range(len(EMOTIONS))),
                                target_names=EMOTIONS, zero_division=0))


if __name__ == "__main__":
    main()
