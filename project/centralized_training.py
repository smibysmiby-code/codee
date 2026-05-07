"""
centralized_training.py
=======================

Phase 2 baseline: pool all 5 device CSV files together, train a single
MLP on the union, and report test accuracy + confusion matrix.

This is the "upper bound" we compare federated learning against -- if
all devices were willing to share their raw data with the UAV (which is
the privacy concern that motivates FL!), we'd train this model.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import config
from model import (
    OffloadingMLP,
    dataframe_to_tensors,
    make_optimizer,
    build_loss,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pool_device_csvs() -> pd.DataFrame:
    """Concatenate all per-device CSVs into a single DataFrame."""
    frames = []
    for dev in range(config.NUM_DEVICES):
        frames.append(pd.read_csv(config.device_csv_path(dev)))
    return pd.concat(frames, ignore_index=True)


def _train_one_epoch(model: nn.Module,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: nn.Module,
                     batch_size: int) -> float:
    """One pass over the training data using simple shuffled mini-batches."""
    model.train()
    n = X.size(0)
    # Random permutation -> shuffled mini-batches (no DataLoader needed).
    perm = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        idx   = perm[start:start + batch_size]
        x_b   = X[idx]
        y_b   = y[idx]
        optimizer.zero_grad()
        logits = model(x_b)
        loss   = loss_fn(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_b.size(0)
    return total_loss / n


def evaluate(model: nn.Module,
             X: torch.Tensor,
             y: torch.Tensor) -> Tuple[float, np.ndarray]:
    """Return (accuracy, confusion-matrix) on the (X, y) tensors."""
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    acc  = float(accuracy_score(y_np, preds))
    cm   = confusion_matrix(y_np, preds, labels=[0, 1])
    return acc, cm


def train_centralized() -> Tuple[OffloadingMLP, float]:
    """Train the centralised baseline and return (model, test_accuracy)."""
    _seed_everything(config.RANDOM_SEED)

    print("[centralized] Pooling per-device CSVs...")
    df = _pool_device_csvs()
    print(f"[centralized] Pooled dataset size: {len(df)} tasks")

    # 80/20 train/test split, stratified on the label column to keep the
    # class balance roughly equal in both splits.
    train_df, test_df = train_test_split(
        df,
        test_size=config.CENTRALIZED_TEST_SPLIT,
        stratify=df[config.LABEL_COLUMN],
        random_state=config.RANDOM_SEED,
    )

    X_train, y_train = dataframe_to_tensors(train_df)
    X_test,  y_test  = dataframe_to_tensors(test_df)

    model     = OffloadingMLP()
    optimizer = make_optimizer(model)
    loss_fn   = build_loss()

    print(f"[centralized] Training for {config.CENTRALIZED_EPOCHS} epochs "
          f"(batch_size={config.BATCH_SIZE})...")
    for epoch in range(1, config.CENTRALIZED_EPOCHS + 1):
        train_loss = _train_one_epoch(
            model, X_train, y_train, optimizer, loss_fn, config.BATCH_SIZE
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == config.CENTRALIZED_EPOCHS:
            test_acc, _ = evaluate(model, X_test, y_test)
            print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"test_acc={test_acc:.4f}")

    test_acc, cm = evaluate(model, X_test, y_test)
    print(f"[centralized] Final test accuracy: {test_acc:.4f}")
    print(f"[centralized] Confusion matrix (rows=truth, cols=pred):")
    print(f"               pred=0   pred=1")
    print(f"   true=0    {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"   true=1    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Persist model so evaluate.py can re-load it without re-training.
    model_path = os.path.join(config.MODEL_DIR, "centralized.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[centralized] Saved model -> {model_path}")

    return model, test_acc


if __name__ == "__main__":
    train_centralized()
