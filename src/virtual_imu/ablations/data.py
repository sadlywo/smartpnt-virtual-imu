"""Recording-disjoint shared inputs and identical supervised center blocks."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from virtual_imu.fir_arx_research.data import UnifiedDataset, LOW, HIGH

LOG = logging.getLogger(__name__)
KEYS = ["dataset_id", "pair_id", "sequence_id", "timestamp_s"]


def support_digest(frame):
    return hashlib.sha256(frame[KEYS].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


class SharedSource:
    """Inspect all role metadata while loading numerical signals only on request."""

    def __init__(self, root, config, protocol):
        self.root, self.config, self.protocol = Path(root), config, protocol
        self.dataset = UnifiedDataset(self.root, config["manifest"], verify_hashes=config.get("verify_hashes", True))
        self.records = self.dataset.specs[protocol]["splits"]
        self.loaded_roles = []
        LOG.info("READS manifest and split identities protocol=%s; SKIPS unrequested role payloads", protocol)

    def load(self, role):
        self.loaded_roles.append(role)
        pieces = []
        for record in self.records[role]:
            if int(record["rows"]) < self.config["max_window"]:
                LOG.info("SKIPS short %s", record["path"])
                continue
            LOG.info("READS protocol=%s role=%s %s", self.protocol, role, record["path"])
            frame = self.dataset.load_record(record)
            keep = len(frame) if role == "train" else len(frame) // self.config["stride"] * self.config["stride"]
            selected = frame.iloc[:keep].copy()
            selected["physical_group_id"] = record["physical_group_id"]
            selected["split"] = role
            pieces.append(selected)
        if not pieces:
            raise ValueError("Empty requested split: " + role)
        return pd.concat(pieces, ignore_index=True)


class CommonWindows:
    def __init__(self, frame, max_window=320, block_size=40, stride=40, stats=None):
        if block_size != stride or min(block_size, max_window) <= 0 or max_window < block_size or (max_window - block_size) % 2:
            raise ValueError("Require stride==block_size and an even centered context expansion")
        self.frame = frame.copy().reset_index(drop=True)
        self.max_window, self.block_size, self.stride = max_window, block_size, stride
        features = self.frame[LOW].to_numpy(np.float64)
        residual = self.frame[HIGH].to_numpy(np.float64) - features
        if not np.isfinite(features).all() or not np.isfinite(residual).all():
            raise ValueError("Nonfinite measured paired signals")
        if stats is None:
            fs, rs = features.std(0), residual.std(0)
            stats = {"feature_mean": features.mean(0).tolist(), "feature_std": np.where(fs < 1e-8, 1., fs).tolist(),
                     "residual_mean": residual.mean(0).tolist(), "residual_std": np.where(rs < 1e-8, 1., rs).tolist()}
        self.stats = stats
        self.x = ((features - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
        self.y = ((residual - stats["residual_mean"]) / stats["residual_std"]).astype(np.float32)
        self.starts, self.anchor_datasets, targets = [], [], []
        half = (max_window - block_size) // 2
        for seq, group in self.frame.groupby("sequence_id", sort=False):
            indices = group.index.to_numpy()
            if not np.all(np.diff(indices) == 1) or group.dataset_id.nunique() != 1:
                raise ValueError("Sequence identity/order must be contiguous")
            times = group.timestamp_s.to_numpy()
            if not np.all(np.diff(times) > 0):
                raise ValueError("Non-increasing timestamps")
            if len(group) < max_window:
                LOG.info("SKIPS short sequence %s", seq)
                continue
            for start in range(int(indices[0]), int(indices[-1]) + 2 - max_window, stride):
                self.starts.append(start)
                self.anchor_datasets.append(str(group.dataset_id.iloc[0]))
                targets.extend(range(start + half, start + half + block_size))
        if not self.starts:
            raise ValueError("No complete common-context anchors")
        self.target_indices = np.array(targets, dtype=np.int64)
        self.target_frame = self.frame.iloc[self.target_indices].reset_index(drop=True)
        if self.target_frame.duplicated(KEYS).any():
            raise ValueError("Primary blocks may not overlap")
        self.anchor_datasets = np.array(self.anchor_datasets)
        self.weights = np.array([1. / np.count_nonzero(self.anchor_datasets == d) for d in self.anchor_datasets])
        self.support_sha256 = support_digest(self.target_frame)

    def __len__(self):
        return len(self.starts)

    def arrays(self, window):
        if not self.block_size <= window <= self.max_window or (window - self.block_size) % 2:
            raise ValueError("Invalid centered context length")
        offset = (self.max_window - window) // 2
        center = (self.max_window - self.block_size) // 2
        x = np.stack([self.x[s+offset:s+offset+window].T for s in self.starts])
        y = np.stack([self.y[s+center:s+center+self.block_size].T for s in self.starts])
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def physical(self, normalized):
        if normalized.shape != (len(self), 6, self.block_size):
            raise ValueError("Wrong prediction block shape")
        residual = normalized.transpose(0, 2, 1).reshape(-1, 6).astype(np.float64)
        residual = residual * self.stats["residual_std"] + self.stats["residual_mean"]
        return self.target_frame[LOW].to_numpy(np.float64) + residual


def balanced_order(data, seed, epoch):
    generator = torch.Generator().manual_seed(int(seed) + int(epoch) * 1000003)
    return torch.multinomial(torch.as_tensor(data.weights, dtype=torch.float64), len(data), True, generator=generator).numpy()


def grouped_score(frame, prediction):
    if len(frame) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("Incomplete/nonfinite prediction support")
    pair_scores = []
    for identity, indices in frame.groupby(["dataset_id", "pair_id"], sort=False).indices.items():
        high = frame.iloc[indices][HIGH].to_numpy(float)
        raw = frame.iloc[indices][LOW].to_numpy(float)
        err = prediction[indices] - high
        baseline = raw - high
        ratios = [np.sqrt(np.mean(err[:, sl] ** 2)) / max(np.sqrt(np.mean(baseline[:, sl] ** 2)), 1e-12) for sl in [slice(0, 3), slice(3, 6)]]
        pair_scores.append({"dataset": identity[0], "score": np.mean(ratios)})
    return float(pd.DataFrame(pair_scores).groupby("dataset").score.mean().mean())


def physical_metrics(frame, prediction):
    """Compute physical component RMSE; pool sample squares before taking roots."""
    if len(prediction) != len(frame):
        raise ValueError("Incomplete metric support")
    results = []
    layouts = [("overall", [], [((), np.arange(len(frame)))]),
               ("dataset", ["dataset_id"], frame.groupby(["dataset_id"], sort=False).indices.items()),
               ("pair", ["dataset_id", "pair_id"], frame.groupby(["dataset_id", "pair_id"], sort=False).indices.items()),
               ("recording", ["dataset_id", "pair_id", "original_sequence_id"], frame.groupby(["dataset_id", "pair_id", "original_sequence_id"], sort=False).indices.items())]
    for aggregation, cols, groups in layouts:
        for identity, indices in groups:
            identity = identity if isinstance(identity, tuple) else (identity,)
            row = dict(aggregation=aggregation, dataset_id="__all__", pair_id="__all__", original_sequence_id="__all__", samples=len(indices))
            row.update(dict(zip(cols, identity)))
            err = prediction[indices] - frame.iloc[indices][HIGH].to_numpy(float)
            for group, sl in [("gyro", slice(0, 3)), ("accel", slice(3, 6))]:
                e = err[:, sl]
                row[group + "_rmse"] = float(np.sqrt(np.mean(e ** 2)))
                row[group + "_vector_rmse"] = float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))
                row[group + "_mae"] = float(np.mean(np.abs(e)))
                row[group + "_bias_norm"] = float(np.linalg.norm(e.mean(0)))
            for index, axis in enumerate(["gx", "gy", "gz", "ax", "ay", "az"]):
                row[axis + "_rmse"] = float(np.sqrt(np.mean(err[:, index] ** 2)))
            results.append(row)
    return results
