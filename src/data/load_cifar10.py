import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_batch(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].astype(np.uint8)
    labels = np.array(batch["labels"], dtype=np.int64)
    # reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels


def load_cifar10_from_raw(raw_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    cifar_dir = raw_dir / "cifar-10-batches-py"
    if not cifar_dir.exists():
        raise FileNotFoundError(f"Expected CIFAR-10 extracted at: {cifar_dir}")

    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        d, l = _load_batch(cifar_dir / f"data_batch_{i}")
        train_data_list.append(d)
        train_labels_list.append(l)

    x_train = np.concatenate(train_data_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)

    x_test, y_test = _load_batch(cifar_dir / "test_batch")

    return (x_train, y_train), (x_test, y_test)


def save_npz_processed(out_dir: Path, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cifar10.npz"
    np.savez_compressed(out_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return out_path


