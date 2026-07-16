from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import random
import re
from typing import Any

import numpy as np
import scipy.signal as sig
import torch



DT = 1.953125e-4
FS = 1.0 / DT
DEFAULT_SPLIT_SEED = 0
DEFAULT_N_TRAIN = 744
DEFAULT_N_VAL = 259
DEFAULT_N_TEST = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def sim_id_from_name(path: str) -> int:
    match = re.search(r"sim_(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot parse simulation id from {path}")
    return int(match.group(1))


def load_metadata(meta_dir: str) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    meta = {}
    for name in sorted(os.listdir(meta_dir)):
        path = os.path.join(meta_dir, name)
        if not os.path.isfile(path) or not name.endswith(".pth"):
            continue
        sid = sim_id_from_name(name)
        meta[sid] = torch.load(path, weights_only=False)
    all_ids = np.array(sorted(meta.keys()), dtype=np.int64)
    return meta, all_ids


def make_train_val_test_split(
    all_ids: np.ndarray,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
    n_test: int = DEFAULT_N_TEST,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_train + n_val + n_test > len(all_ids):
        raise ValueError("Requested split exceeds available trajectories.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(all_ids)
    train_ids = np.sort(perm[:n_train])
    val_ids = np.sort(perm[n_train:n_train + n_val])
    test_ids = np.sort(perm[n_train + n_val:n_train + n_val + n_test])
    return train_ids, val_ids, test_ids


def compute_label_log_stats_from_ids(
    dct_meta: dict[int, dict[str, Any]],
    traj_ids: np.ndarray,
    w_size: int,
    max_steps: int,
) -> tuple[float, float]:
    labels = []
    for tid in traj_ids:
        onset_ts = int(dct_meta[int(tid)]["onset_ts"])
        n_adm = min(max_steps, onset_ts - w_size)
        if n_adm <= 0:
            continue
        labels.append(torch.arange(1, n_adm + 1, dtype=torch.float32))
    if not labels:
        raise RuntimeError("No admissible labels for the requested split.")
    labels = torch.cat(labels)
    return torch.log(labels).mean().item(), torch.log(labels).std().item()


def load_experimental_onsets(path: str) -> dict[int, int]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): int(v) for k, v in raw.items()}


def load_experimental_run(path: str) -> torch.Tensor:

    arr = torch.load(path, mmap=True, weights_only=False)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    arr = sig.decimate(arr, 10, ftype="fir", zero_phase=True, axis=0)
    return torch.from_numpy(arr).to(torch.float32)


def steps_to_seconds(x: np.ndarray | torch.Tensor) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * DT


@dataclass
class SyntheticOnsetConfig:
    alpha: float = 0.03
    persistence_samples: int = 4000
    persistence_fraction: float = 0.15
    pers_adjust: int = 1000

    # Legacy adaptive knobs kept for backward-compatible parsing only.
    win_start: int | None = None
    win_bef_aft_ons: int | None = None

    def short_name(self) -> str:
        return (
            f"thresh_a{str(self.alpha).replace('.', 'p')}_"
            f"pq{self.persistence_samples}_"
            f"pa{self.pers_adjust}"
        )

    def resolved_win_start(self) -> int:
        if self.win_start is not None:
            return int(self.win_start)
        if self.win_bef_aft_ons is not None:
            return int(self.win_bef_aft_ons)
        return int(self.persistence_samples)

    def resolved_win_bef_aft_ons(self) -> int:
        return self.resolved_win_start()


def threshold_persistence_onset_from_trace(
    p_sens: np.ndarray,
    cfg: SyntheticOnsetConfig,
) -> int | None:
    p_sens = np.asarray(p_sens, dtype=np.float64)
    if p_sens.ndim == 1:
        p_sens = p_sens[:, None]
    if p_sens.ndim != 2:
        raise ValueError(f"Expected saved run with shape [T, C], got {p_sens.shape}")

    p_max_time = np.abs(p_sens).max(axis=1)
    n = int(p_max_time.size)
    pers_quantile = int(cfg.persistence_samples)
    pers_adjust = int(cfg.pers_adjust)

    if pers_quantile <= 0 or pers_adjust <= 0:
        raise ValueError("persistence_samples and pers_adjust must be positive")
    if n < pers_quantile:
        return None

    p_stable = float(np.quantile(np.abs(p_max_time[:pers_quantile]), 0.9))
    p_unstable = float(np.quantile(np.abs(p_max_time[-pers_quantile:]), 0.9))
    state_diff = p_unstable - p_stable
    p_thresh = p_stable + float(cfg.alpha) * state_diff

    msk = (p_max_time > p_thresh).astype(np.int64)
    sliding = np.convolve(msk, np.ones(pers_quantile, dtype=np.int64), mode="valid")
    onset_candidates = np.flatnonzero(sliding > float(cfg.persistence_fraction) * pers_quantile)
    if onset_candidates.size == 0:
        return None
    onset = int(onset_candidates[0])
    if onset == 0:
        return None

    begin = max(0, onset - 50000)
    onset_settled = False
    trials = 0

    while not onset_settled:
        if trials > 5000:
            return None
        if onset < 7000 or onset > n:
            return None

        stab_hi = min(begin + pers_adjust, n)
        before_lo = max(onset - pers_adjust, 0)
        after_hi = min(onset + pers_adjust, n)
        if stab_hi <= begin or onset <= before_lo or after_hi <= onset:
            return None

        std_stab = float(p_max_time[begin:stab_hi].std())
        std_before_onset = float(p_max_time[before_lo:onset].std())
        std_after_onset = float(p_max_time[onset:after_hi].std())

        if std_before_onset > std_stab * 2.5:
            onset -= 5
            begin = max(0, onset - 50000)
            trials += 1
        elif std_after_onset < std_stab * 2.5:
            onset += 5
            begin = max(0, onset - 50000)
            trials += 1
        else:
            onset_settled = True

    return int(onset)


def relabel_onset_from_saved_run(
    p_sens: np.ndarray,
    cfg: SyntheticOnsetConfig,
) -> int | None:
    return threshold_persistence_onset_from_trace(p_sens, cfg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
