

# Training script. 
# This script is long mainly because of training logs printing. The training loop code is not extensive. 
# Works with preprocessing module. 

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import trans_reg_model
from preprocess_mod import WindowDataset
from rebuttal_analysis.common import (
    DT,
    ensure_dir,
    load_experimental_onsets,
    load_experimental_run,
    load_metadata,
    set_seed,
    write_json,
)


ACTIVE_SENSOR_IDXS = torch.tensor([0, 4, 8], dtype=torch.long)
MAX_STEPS = 10240
WINDOW_STEPS = 80
SAMPLES_PER_BIN = 7


@dataclass
class Config:
    inp_seq_len: int = 1500
    hidden_dim: int = 600
    dropout: float = 0.06
    batch_size: int = 1024
    lr: float = 0.5
    weight_decay: float = 8.22e-4
    n_heads: int = 3
    num_lay_trans: int = 5
    regression_head: int = 700
    num_layers: int = 1
    beta2: float = 0.9632
    grad_clip_norm: float = 2.0
    warmup_steps: int = 2500
    warmup_step_offset: int = 0
    max_epochs: int = 6000
    min_epochs_before_validation: int = 25
    patience: int = 1500
    min_delta: float = 0.0
    min_epochs_before_early_stop: int = 5000
    use_amp_train: bool = False
    use_amp_eval: bool = True
    inp_buff: int = 10240
    num_workers_train: int = 4
    num_workers_val: int = 3
    samples_per_epoch_train: int = 16384
    samples_per_epoch_val: int = 16384
    windows_per_traj_train: int = 32
    windows_per_traj_val: int = 8
    n_traj_train: int = 900
    n_traj_burst_train: int = 25
    n_traj_burst_val: int = 259
    requested_val_size: int | None = None
    requested_test_size: int | None = None
    exact_n_sensors: int = 3
    finetune_drop_to_min_sensors: int = 3
    finetune_drop_to_max_sensors: int = 3
    runs_dir: str = "data/runs3"
    seed: int = 1337


def worker_init_fn(worker_id: int) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(2)


def compute_uniform_label_log_stats(inp_buff: int) -> tuple[float, float]:
    """Normalize against the dataset's uniform lead-time support: 1..inp_buff steps."""
    if inp_buff < 1:
        raise ValueError("inp_buff must be >= 1 for label normalization.")
    labs = torch.arange(1, int(inp_buff) + 1, dtype=torch.float64)
    log_labs = torch.log(labs)
    return log_labs.mean().item(), log_labs.std(unbiased=True).item()

def split_trajectory_ids(dct_meta: dict[int, dict[str, Any]], cfg: Config) -> dict[str, np.ndarray]:
    all_ids = np.array(sorted(dct_meta.keys()), dtype=np.int64)
    if cfg.n_traj_train > len(all_ids):
        raise ValueError("Requested training pool exceeds available trajectories.")

    
    rng = np.random.default_rng(seed=0)
    train_ids = np.sort(rng.choice(all_ids, cfg.n_traj_train, replace=False))
    remaining_ids = np.setdiff1d(all_ids, train_ids, assume_unique=True)
    remaining_count = int(len(remaining_ids))

    if cfg.requested_val_size is None and cfg.requested_test_size is None:
        requested_val_size = remaining_count
        requested_test_size = 0
    else:
        if cfg.requested_val_size is None:
            requested_test_size = max(0, min(int(cfg.requested_test_size or 0), remaining_count))
            requested_val_size = remaining_count - requested_test_size
        else:
            requested_val_size = max(0, min(int(cfg.requested_val_size), remaining_count))
            requested_test_size = remaining_count - requested_val_size
            if cfg.requested_test_size is not None:
                requested_test_size = max(0, min(int(cfg.requested_test_size), remaining_count - requested_val_size))
                requested_test_size += remaining_count - requested_val_size - requested_test_size

    if requested_val_size > 0:
        val_ids = np.sort(rng.choice(remaining_ids, requested_val_size, replace=False))
    else:
        val_ids = np.array([], dtype=np.int64)
    test_ids = np.setdiff1d(remaining_ids, val_ids, assume_unique=True)
    return {
        "all_ids": all_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def subset_metadata_by_ids(
    dct_meta: dict[int, dict[str, Any]],
    ids: np.ndarray,
) -> dict[int, dict[str, Any]]:
    return {int(tid): dct_meta[int(tid)] for tid in ids.tolist()}


def count_eligible_ids(dct_meta: dict[int, dict[str, Any]], ids: np.ndarray, cfg: Config) -> int:
    return int(
        sum(int(dct_meta[int(tid)]["onset_ts"]) > cfg.inp_buff + cfg.inp_seq_len for tid in ids)
    )


def compute_pool_sizes_like_original(dct_meta: dict[int, dict[str, Any]], cfg: Config) -> dict[str, int]:
    split_ids = split_trajectory_ids(dct_meta, cfg)
    all_ids = split_ids["all_ids"]
    train_ids = split_ids["train_ids"]
    val_ids = split_ids["val_ids"]
    test_ids = split_ids["test_ids"]
    eligible_train = count_eligible_ids(dct_meta, train_ids, cfg)
    eligible_val = count_eligible_ids(dct_meta, val_ids, cfg)
    eligible_test = count_eligible_ids(dct_meta, test_ids, cfg)
    if eligible_train < 1 or eligible_val < 1:
        raise ValueError("No eligible train/validation trajectories remain after applying onset/window constraints.")
    return {
        "n_total": int(len(all_ids)),
        "n_traj_train": int(len(train_ids)),
        "val_pool_size": int(len(val_ids)),
        "test_pool_size": int(len(test_ids)),
        "eligible_train": eligible_train,
        "eligible_val": eligible_val,
        "eligible_test": eligible_test,
        "n_traj_burst_train": min(int(cfg.n_traj_burst_train), eligible_train),
        "n_traj_burst_val": min(int(cfg.n_traj_burst_val), int(len(val_ids)), eligible_val),
        "n_traj_burst_test": min(int(len(test_ids)), eligible_test),
        "requested_val_size": int(len(val_ids)),
        "requested_test_size": int(len(test_ids)),
    }


def make_loader(dct_meta: dict[int, dict[str, Any]], cfg: Config, train: bool, pool_sizes: dict[str, int]) -> DataLoader:
    dataset = WindowDataset(
        dct_meta=dct_meta,
        w_size=cfg.inp_seq_len,
        n_sens_min=1,
        n_sens_max=12,
        n_epoch=cfg.max_epochs,
        samples_per_epoch=cfg.samples_per_epoch_train if train else cfg.samples_per_epoch_val,
        inp_buff=cfg.inp_buff,
        n_traj_train=pool_sizes["n_traj_train"],
        n_traj_burst=pool_sizes["n_traj_burst_train"] if train else pool_sizes["n_traj_burst_val"],
        windows_per_traj=cfg.windows_per_traj_train if train else cfg.windows_per_traj_val,
        seed=cfg.seed,
        train=train,
        dynamic_sensor_sampling=False,
        biased_sampling=False,
        i_sens_val=cfg.exact_n_sensors - 1,
        exact_n_sensors=cfg.exact_n_sensors,
        runs_dir=cfg.runs_dir,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers_train if train else cfg.num_workers_val,
        pin_memory=True,
        prefetch_factor=30 if train else 15,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )


def make_subset_loader(
    dct_meta_subset: dict[int, dict[str, Any]],
    cfg: Config,
    *,
    train: bool,
    n_traj_burst: int,
) -> DataLoader:
    subset_pool_sizes = {
        "n_traj_train": int(len(dct_meta_subset)) if train else 0,
        "n_traj_burst_train": int(min(n_traj_burst, len(dct_meta_subset))) if train else 0,
        "n_traj_burst_val": int(min(n_traj_burst, len(dct_meta_subset))) if not train else 0,
    }
    return make_loader(dct_meta_subset, cfg, train=train, pool_sizes=subset_pool_sizes)


def make_model(cfg: Config, device: torch.device) -> trans_reg_model:
    return trans_reg_model(
        seq_dim=cfg.inp_seq_len,
        inp_embed_dim=24,
        embed_dim=24,
        n_heads=cfg.n_heads,
        num_lay_trans=cfg.num_lay_trans,
        regression_head=cfg.regression_head,
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)


def sample_sensor_subset(
    sens_ind_msk: torch.Tensor,
    *,
    rng: np.random.Generator,
    min_active: int,
    max_active: int,
) -> torch.Tensor:
    sens_new = sens_ind_msk.clone()
    for b in range(sens_new.size(0)):
        active = torch.nonzero(sens_new[b], as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        k_hi = min(max_active, int(active.numel()))
        k_lo = min(min_active, k_hi)
        k_keep = int(rng.integers(k_lo, k_hi + 1))
        keep_idx = rng.choice(active.cpu().numpy(), size=k_keep, replace=False)
        keep = torch.zeros_like(sens_new[b])
        keep[torch.as_tensor(keep_idx, dtype=torch.long)] = True
        sens_new[b] = keep
    return sens_new


def lr_lambda(step: int, d_model: int, warmup_steps: int, warmup_step_offset: int) -> float:
    step = max(step, 1) + warmup_step_offset
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def summarize_support_s(values_s: np.ndarray, prefix: str) -> dict[str, float]:
    values_s = np.asarray(values_s, dtype=np.float64)
    values_s = values_s[np.isfinite(values_s)]
    if values_s.size == 0:
        return {
            f"{prefix}_min_s": float("nan"),
            f"{prefix}_p05_s": float("nan"),
            f"{prefix}_p25_s": float("nan"),
            f"{prefix}_p50_s": float("nan"),
            f"{prefix}_p75_s": float("nan"),
            f"{prefix}_p95_s": float("nan"),
            f"{prefix}_max_s": float("nan"),
        }
    return {
        f"{prefix}_min_s": float(np.min(values_s)),
        f"{prefix}_p05_s": float(np.percentile(values_s, 5.0)),
        f"{prefix}_p25_s": float(np.percentile(values_s, 25.0)),
        f"{prefix}_p50_s": float(np.percentile(values_s, 50.0)),
        f"{prefix}_p75_s": float(np.percentile(values_s, 75.0)),
        f"{prefix}_p95_s": float(np.percentile(values_s, 95.0)),
        f"{prefix}_max_s": float(np.max(values_s)),
    }


def evaluate_loader(
    cfg,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lab_mean_log: float,
    lab_std_log: float,
    use_amp_eval: bool = True,
) -> dict[str, float]:
    model.eval()
    use_amp = device.type == "cuda" and use_amp_eval
    loss_sum = 0.0
    mae_z_sum = 0.0
    mae_s_sum = 0.0
    pred_std_sum = 0.0
    targ_std_sum = 0.0
    n_batches = 0
    pred_s_all: list[np.ndarray] = []
    true_s_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, sens_ind_msk, labels_log in loader:
            x = torch.concat([x, sens_ind_msk[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
            x = x.to(device, dtype=torch.float32, non_blocking=True).contiguous()
            labels_log = labels_log.to(device, dtype=torch.float32, non_blocking=True)
            labels_z = (labels_log - lab_mean_log) / lab_std_log
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_z = model(x)
                #loss = model.loss_fun(pred_z, labels_z, lab_mean_log, lab_std_log, cfg.inp_buff)
                loss = model.loss_fun(pred_z, labels_z)
            pred_log = pred_z.float() * lab_std_log + lab_mean_log
            pred_s = torch.exp(pred_log)
            true_s = torch.exp(labels_log)
            pred_s_all.append((pred_s.detach().cpu().numpy().astype(np.float64) * DT))
            true_s_all.append((true_s.detach().cpu().numpy().astype(np.float64) * DT))
            loss_sum += float(loss.item())
            mae_z_sum += float(torch.abs(pred_z.float() - labels_z).mean().item())
            mae_s_sum += float(torch.abs(pred_s - true_s).mean().item() * DT)
            pred_std_sum += float(pred_z.float().std().item())
            targ_std_sum += float(labels_z.float().std().item())
            n_batches += 1
    pred_all = np.concatenate(pred_s_all) if pred_s_all else np.asarray([])
    true_all = np.concatenate(true_s_all) if true_s_all else np.asarray([])
    pred_support = summarize_support_s(pred_all, "pred")
    true_support = summarize_support_s(true_all, "true")
    return {
        "loss": loss_sum / max(n_batches, 1),
        "mae_z": mae_z_sum / max(n_batches, 1),
        "mae_s": mae_s_sum / max(n_batches, 1),
        "pred_std_z": pred_std_sum / max(n_batches, 1),
        "targ_std_z": targ_std_sum / max(n_batches, 1),
        "pred_std_s": float(np.std(pred_all)) if pred_all.size > 0 else float("nan"),
        "targ_std_s": float(np.std(true_all)) if true_all.size > 0 else float("nan"),
        **pred_support,
        **true_support,
    }


def make_balanced_distances(max_steps: int = MAX_STEPS, window_steps: int = WINDOW_STEPS, samples_per_bin: int = SAMPLES_PER_BIN) -> np.ndarray:
    dists: list[int] = []
    for j in range(max_steps // window_steps):
        lo = j * window_steps + 1
        hi = (j + 1) * window_steps
        xs = np.linspace(lo, hi, samples_per_bin, endpoint=True)
        xs = np.unique(np.clip(np.round(xs).astype(np.int64), lo, hi))
        dists.extend(xs.tolist())
    d = np.asarray(dists, dtype=np.int64)
    return d[(d >= 1) & (d <= max_steps)]


def build_standardized_window(traj_3ch: torch.Tensor, onset: int, k_end: int, w_size: int, inp_buff: int) -> torch.Tensor:
    x_raw = traj_3ch[k_end - w_size : k_end, :]
    traj_mean = torch.zeros(12, dtype=torch.float32)
    traj_std = torch.ones(12, dtype=torch.float32)
    norm_seg = traj_3ch[onset - inp_buff - w_size : k_end, :]
    traj_mean[ACTIVE_SENSOR_IDXS] = norm_seg.mean(dim=0)
    traj_std[ACTIVE_SENSOR_IDXS] = norm_seg.std(dim=0).clamp_min(1e-6)
    x_full = torch.zeros((w_size, 12), dtype=torch.float32)
    x_full[:, ACTIVE_SENSOR_IDXS] = x_raw
    return (x_full - traj_mean) / traj_std


def evaluate_experiment(
    model: torch.nn.Module,
    device: torch.device,
    exp_run_dir: str,
    exp_onset_json: str,
    lab_mean_log: float,
    lab_std_log: float,
    w_size: int,
    inp_buff: int,
    batch_size: int,
    use_amp_eval: bool = True,
) -> dict[str, Any]:
    onset_map = load_experimental_onsets(exp_onset_json)
    fixed_mask = torch.zeros(12, dtype=torch.bool)
    fixed_mask[ACTIVE_SENSOR_IDXS] = True
    all_true_s: list[np.ndarray] = []
    all_pred_s: list[np.ndarray] = []
    run_metrics = []
    dists = make_balanced_distances()
    use_amp = device.type == "cuda" and use_amp_eval

    model.eval()
    with torch.no_grad():
        for run_id, onset in sorted(onset_map.items()):
            traj = load_experimental_run(os.path.join(exp_run_dir, f"ramp_{run_id}.pt"))
            windows = []
            true_steps = []
            for dist in dists:
                k_end = onset - int(dist)
                if k_end <= w_size or onset <= inp_buff + w_size or k_end > traj.shape[0]:
                    continue
                windows.append(build_standardized_window(traj, onset, k_end, w_size, inp_buff))
                true_steps.append(dist)
            if not windows:
                continue
            preds = []
            for i in range(0, len(windows), batch_size):
                xb = torch.stack(windows[i : i + batch_size], dim=0)
                mb = fixed_mask[None, None, :].expand(xb.size(0), xb.size(1), -1)
                xb = torch.concat([xb, mb.to(dtype=xb.dtype)], dim=-1).to(device, dtype=torch.float32, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = model(xb)
                pred_s = torch.exp(pred_z.float() * lab_std_log + lab_mean_log).cpu().numpy() * DT
                preds.append(pred_s)
            pred_s = np.concatenate(preds)
            true_s = np.asarray(true_steps, dtype=np.float64) * DT
            run_mae = float(np.mean(np.abs(pred_s - true_s)))
            run_metrics.append({"run_id": int(run_id), "mae_s": run_mae, "n_windows": int(len(true_s))})
            all_true_s.append(true_s)
            all_pred_s.append(pred_s)

    if not all_true_s:
        return {
            "overall_mae_s": float("nan"),
            "overall_pred_std_s": float("nan"),
            "run_metrics": [],
        }
    y_true = np.concatenate(all_true_s)
    y_pred = np.concatenate(all_pred_s)
    return {
        "overall_mae_s": float(np.mean(np.abs(y_pred - y_true))),
        "overall_pred_std_s": float(np.std(y_pred)),
        "n_runs": int(len(run_metrics)),
        "n_windows": int(len(y_true)),
        "run_metrics": run_metrics,
        **summarize_support_s(y_pred, "pred"),
        **summarize_support_s(y_true, "true"),
    }


def load_init_state(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    return raw


def resolve_monitor_dir(out_dir: str, monitor_dir: str | None) -> str:
    if monitor_dir:
        return monitor_dir
    env_dir = os.environ.get("THERMAC_MONITOR_DIR")
    if env_dir:
        return env_dir
    return out_dir


def write_training_monitor(
    monitor_dir: str,
    tag: str,
    *,
    state: str,
    summary_path: str,
    checkpoint_path: str,
    best_epoch: int,
    best_val_mae_z: float | None,
    latest_history: dict[str, float] | None,
    history_length: int,
    training_in_progress: bool,
    test_metrics: dict[str, Any] | None = None,
    exp_metrics: dict[str, Any] | None = None,
) -> None:
    ensure_dir(monitor_dir)
    payload: dict[str, Any] = {
        "tag": tag,
        "state": state,
        "summary_path": os.path.abspath(summary_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_mae_z": best_val_mae_z,
        "history_length": int(history_length),
        "training_in_progress": bool(training_in_progress),
        "updated_at_unix": time.time(),
    }
    if latest_history is not None:
        payload["latest_history"] = latest_history
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    if exp_metrics is not None:
        payload["experimental_transfer"] = exp_metrics
    write_json(os.path.join(monitor_dir, f"{tag}_monitor.json"), payload)


def run_training(
    meta_dir: str,
    out_dir: str,
    exp_run_dir: str,
    exp_onset_json: str,
    device_name: str,
    cfg: Config,
    tag: str,
    init_checkpoint: str | None,
    monitor_dir: str | None,
) -> None:
    """
    Full training function. Splits the data into train/val/test sets, then load it then trains the model with logs.
    """
    ensure_dir(out_dir)
    monitor_dir = resolve_monitor_dir(out_dir, monitor_dir)
    ensure_dir(monitor_dir)
    set_seed(cfg.seed)
    device = torch.device(device_name)
    dct_meta, _ = load_metadata(meta_dir)
    split_ids = split_trajectory_ids(dct_meta, cfg)
    pool_sizes = compute_pool_sizes_like_original(dct_meta, cfg)
    train_meta = subset_metadata_by_ids(dct_meta, split_ids["train_ids"])
    val_meta = subset_metadata_by_ids(dct_meta, split_ids["val_ids"])
    test_meta = subset_metadata_by_ids(dct_meta, split_ids["test_ids"])
    lab_mean_log, lab_std_log = compute_uniform_label_log_stats(cfg.inp_buff)
    train_loader = make_subset_loader(train_meta, cfg, train=True, n_traj_burst=pool_sizes["n_traj_burst_train"])
    val_loader = make_subset_loader(val_meta, cfg, train=False, n_traj_burst=pool_sizes["n_traj_burst_val"])
    test_loader = make_subset_loader(test_meta, cfg, train=False, n_traj_burst=pool_sizes["n_traj_burst_val"])
    if pool_sizes["n_traj_burst_test"] > 0:
        test_loader = make_subset_loader(test_meta, cfg, train=False, n_traj_burst=pool_sizes["n_traj_burst_test"])

    model = make_model(cfg, device)
    if init_checkpoint:
        model.load_state_dict(load_init_state(init_checkpoint, device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, cfg.beta2),
        eps=1e-9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, model.embed_dim, cfg.warmup_steps, cfg.warmup_step_offset),
    )

    best_val_mae_z = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_ckpt = os.path.join(out_dir, f"{tag}_best.pth")
    summary_path = os.path.join(out_dir, f"{tag}_summary.json")
    history: list[dict[str, float]] = []
    use_amp_train = device.type == "cuda" and cfg.use_amp_train

    write_json(
        summary_path,
        {
            "tag": tag,
            "meta_dir": os.path.abspath(meta_dir),
            "device": str(device),
            "init_checkpoint": os.path.abspath(init_checkpoint) if init_checkpoint else None,
            "best_epoch": int(best_epoch),
            "best_val_mae_z": None,
            "label_log_mean": float(lab_mean_log),
            "label_log_std": float(lab_std_log),
            "train_config": asdict(cfg),
            "pool_sizes": pool_sizes,
            "history": history,
            "checkpoint_path": os.path.abspath(best_ckpt),
            "training_in_progress": True,
            "test_metrics": None,
            "experimental_transfer": {},
        },
    )
    write_training_monitor(
        monitor_dir,
        tag,
        state="starting",
        summary_path=summary_path,
        checkpoint_path=best_ckpt,
        best_epoch=best_epoch,
        best_val_mae_z=None,
        latest_history=None,
        history_length=0,
        training_in_progress=True,
        test_metrics=None,
    )

    for epoch in range(cfg.max_epochs):
        model.train()
        loss_sum = 0.0
        mae_z_sum = 0.0
        pred_std_sum = 0.0
        targ_std_sum = 0.0
        n_batches = 0

        for batch_idx, (x, sens_ind_msk, labels_log) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            # Training/validation/test all use the exact-3 sensor regime.
            rng = np.random.default_rng(cfg.seed + 100000 * epoch + batch_idx)
            sens_ind_msk = sample_sensor_subset(
                sens_ind_msk,
                rng=rng,
                min_active=cfg.finetune_drop_to_min_sensors,
                max_active=cfg.finetune_drop_to_max_sensors,
            )
            x = x * sens_ind_msk[:, None, :].to(dtype=x.dtype)

            x = torch.concat([x, sens_ind_msk[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
            x = x.to(device, dtype=torch.float32, non_blocking=True).contiguous()
            labels_log = labels_log.to(device, dtype=torch.float32, non_blocking=True)
            labels_z = (labels_log - lab_mean_log) / lab_std_log
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=use_amp_train,
            ):
                pred_z = model(x)
                loss = model.loss_fun(pred_z, labels_z)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            loss_sum += float(loss.item())
            mae_z_sum += float(torch.abs(pred_z.float() - labels_z).mean().item())
            pred_std_sum += float(pred_z.float().std().item())
            targ_std_sum += float(labels_z.float().std().item())
            n_batches += 1

        train_metrics = {
            "loss": loss_sum / max(n_batches, 1),
            "mae_z": mae_z_sum / max(n_batches, 1),
            "pred_std_tr": pred_std_sum / max(n_batches, 1),
            "targ_std_tr": targ_std_sum / max(n_batches, 1),
        }
        validation_active = epoch >= cfg.min_epochs_before_validation

        # getting logs information about model performance on validation data
        if validation_active:
            val_metrics = evaluate_loader(cfg, model, val_loader, device, lab_mean_log, lab_std_log, cfg.use_amp_eval)
            history.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "validation_active": True,
                    **train_metrics,
                    "val_loss": val_metrics["loss"],
                    "val_mae_z": val_metrics["mae_z"],
                    "val_mae_s": val_metrics["mae_s"],
                    "val_pred_std_z": val_metrics["pred_std_z"],
                    "val_targ_std_z": val_metrics["targ_std_z"],
                    "val_pred_std_s": val_metrics["pred_std_s"],
                    "val_targ_std_s": val_metrics["targ_std_s"],
                    "val_pred_min_s": val_metrics["pred_min_s"],
                    "val_pred_p05_s": val_metrics["pred_p05_s"],
                    "val_pred_p25_s": val_metrics["pred_p25_s"],
                    "val_pred_p50_s": val_metrics["pred_p50_s"],
                    "val_pred_p75_s": val_metrics["pred_p75_s"],
                    "val_pred_p95_s": val_metrics["pred_p95_s"],
                    "val_pred_max_s": val_metrics["pred_max_s"],
                    "val_true_max_s": val_metrics["true_max_s"],
                }
            )

            print(
                f"[{tag}] epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_mae_z={train_metrics['mae_z']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mae_z={val_metrics['mae_z']:.6f} "
                f"val_mae_s={val_metrics['mae_s']:.6f} "
                f"best_val_mae_z={best_val_mae_z if math.isfinite(best_val_mae_z) else float('nan'):.6f} "
                f"pred_std_tr_z={train_metrics['pred_std_tr']:.4f} "
                f"targ_std_tr_z={train_metrics['targ_std_tr']:.4f} "
                f"pred_std_val_z={val_metrics['pred_std_z']:.4f} "
                f"targ_std_val_z={val_metrics['targ_std_z']:.4f} "
                f"pred_std_val_s={val_metrics['pred_std_s']:.4f} "
                f"targ_std_val_s={val_metrics['targ_std_s']:.4f} "
                f"val_pred_min_s={val_metrics['pred_min_s']:.4f} "
                f"val_pred_p05_s={val_metrics['pred_p05_s']:.4f} "
                f"val_pred_p25_s={val_metrics['pred_p25_s']:.4f} "
                f"val_pred_p50_s={val_metrics['pred_p50_s']:.4f} "
                f"val_pred_p75_s={val_metrics['pred_p75_s']:.4f} "
                f"val_pred_p95_s={val_metrics['pred_p95_s']:.4f} "
                f"val_pred_max_s={val_metrics['pred_max_s']:.4f} "
                f"val_true_max_s={val_metrics['true_max_s']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"patience={patience_counter}/{cfg.patience} "
                f"early_stop_active={epoch >= cfg.min_epochs_before_early_stop}",
                flush=True,
            )

            # getting logs information about model performance on experimental data
            if val_metrics["mae_z"] < best_val_mae_z - cfg.min_delta:
                best_val_mae_z = float(val_metrics["mae_z"])
                best_epoch = int(epoch)
                patience_counter = 0

                exp_metrics_best = evaluate_experiment(
                    model=model,
                    device=device,
                    exp_run_dir=exp_run_dir,
                    exp_onset_json=exp_onset_json,
                    lab_mean_log=lab_mean_log,
                    lab_std_log=lab_std_log,
                    w_size=cfg.inp_seq_len,
                    inp_buff=cfg.inp_buff,
                    batch_size=cfg.batch_size,
                    use_amp_eval=cfg.use_amp_eval,
                )
                experimental_mae_s = exp_metrics_best.get('overall_mae_s', float('nan'))
                experimental_pred_std_s = exp_metrics_best.get('overall_pred_std_s', float('nan'))
                print(
                    f"[{tag}] new best checkpoint at epoch {epoch}: "
                    f"mae_z={best_val_mae_z:.6f} "
                    f"experimental_mae_s={experimental_mae_s:.6f} "
                    f"experimental_pred_std_s={experimental_pred_std_s:.6f}",
                    f"experimental_pred_min_s={exp_metrics_best.get('pred_min_s', float('nan')):.4f} "
                    f"experimental_pred_p05_s={exp_metrics_best.get('pred_p05_s', float('nan')):.4f} "
                    f"experimental_pred_p25_s={exp_metrics_best.get('pred_p25_s', float('nan')):.4f} "
                    f"experimental_pred_p50_s={exp_metrics_best.get('pred_p50_s', float('nan')):.4f} "
                    f"experimental_pred_p75_s={exp_metrics_best.get('pred_p75_s', float('nan')):.4f} "
                    f"experimental_pred_p95_s={exp_metrics_best.get('pred_p95_s', float('nan')):.4f} "
                    f"experimental_pred_max_s={exp_metrics_best.get('pred_max_s', float('nan')):.4f}",
                    flush=True,
                )

                best_ckpt_save = os.path.join(out_dir, f"{tag}_{experimental_mae_s}_{experimental_pred_std_s}.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "lab_mean_log": lab_mean_log,
                        "lab_std_log": lab_std_log,
                        "train_config": asdict(cfg),
                        "tag": tag,
                    },
                    best_ckpt_save,
                )
            else:
                # we do not run training forever, limited number of epochs for which no improvement has been seen before stopping the training 
                patience_counter += 1
        else:
            history.append(
                {
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "validation_active": False,
                    **train_metrics,
                }
            )
            print(
                f"[{tag}] epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_mae_z={train_metrics['mae_z']:.6f} "
                f"pred_std_tr_z={train_metrics['pred_std_tr']:.4f} "
                f"targ_std_tr_z={train_metrics['targ_std_tr']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"validation_active=False "
                f"validation_starts_at={cfg.min_epochs_before_validation}",
                flush=True,
            )

        write_json(
            summary_path,
            {
                "tag": tag,
                "meta_dir": os.path.abspath(meta_dir),
                "device": str(device),
                "init_checkpoint": os.path.abspath(init_checkpoint) if init_checkpoint else None,
                "best_epoch": int(best_epoch),
                "best_val_mae_z": float(best_val_mae_z) if math.isfinite(best_val_mae_z) else None,
                "label_log_mean": float(lab_mean_log),
                "label_log_std": float(lab_std_log),
                "train_config": asdict(cfg),
                "pool_sizes": pool_sizes,
                "history": history,
                "checkpoint_path": os.path.abspath(best_ckpt),
                "training_in_progress": True,
                "test_metrics": None,
                "experimental_transfer": {},
            },
        )
        write_training_monitor(
            monitor_dir,
            tag,
            state="running",
            summary_path=summary_path,
            checkpoint_path=best_ckpt,
            best_epoch=best_epoch,
            best_val_mae_z=float(best_val_mae_z) if math.isfinite(best_val_mae_z) else None,
            latest_history=history[-1] if history else None,
            history_length=len(history),
            training_in_progress=True,
            test_metrics=None,
        )

        if validation_active and epoch >= cfg.min_epochs_before_early_stop and patience_counter >= cfg.patience:
            print(f"[{tag}] early stopping at epoch {epoch}", flush=True)
            break

    if not os.path.exists(best_ckpt):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "lab_mean_log": lab_mean_log,
                "lab_std_log": lab_std_log,
                "train_config": asdict(cfg),
                "tag": tag,
            },
            best_ckpt,
        )

    saved = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"] if "model_state_dict" in saved else saved)
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate_loader(cfg, model, test_loader, device, lab_mean_log, lab_std_log, cfg.use_amp_eval)
    exp_metrics = evaluate_experiment(
        model=model,
        device=device,
        exp_run_dir=exp_run_dir,
        exp_onset_json=exp_onset_json,
        lab_mean_log=lab_mean_log,
        lab_std_log=lab_std_log,
        w_size=cfg.inp_seq_len,
        inp_buff=cfg.inp_buff,
        batch_size=cfg.batch_size,
        use_amp_eval=cfg.use_amp_eval,
    )
    write_json(
        summary_path,
        {
            "tag": tag,
            "meta_dir": os.path.abspath(meta_dir),
            "device": str(device),
            "init_checkpoint": os.path.abspath(init_checkpoint) if init_checkpoint else None,
            "best_epoch": int(best_epoch),
            "best_val_mae_z": float(best_val_mae_z),
            "label_log_mean": float(lab_mean_log),
            "label_log_std": float(lab_std_log),
            "train_config": asdict(cfg),
            "pool_sizes": pool_sizes,
            "history": history,
            "checkpoint_path": os.path.abspath(best_ckpt),
            "training_in_progress": False,
            "test_metrics": test_metrics,
            "experimental_transfer": exp_metrics,
        },
    )
    write_training_monitor(
        monitor_dir,
        tag,
        state="finished",
        summary_path=summary_path,
        checkpoint_path=best_ckpt,
        best_epoch=best_epoch,
        best_val_mae_z=float(best_val_mae_z) if math.isfinite(best_val_mae_z) else None,
        latest_history=history[-1] if history else None,
        history_length=len(history),
        training_in_progress=False,
        test_metrics=test_metrics,
        exp_metrics=exp_metrics,
    )
    final_msg = (
        f"[{tag}] finished: best_epoch={best_epoch}, "
        f"best_val_mae_z={(best_val_mae_z if math.isfinite(best_val_mae_z) else float('nan')):.6f}"
    )
    if test_metrics is not None:
        final_msg += f", test_mae_s={test_metrics['mae_s']:.6f}"
    final_msg += (
        f", experimental_mae_s={exp_metrics.get('overall_mae_s', float('nan')):.6f}"
        f", experimental_pred_std_s={exp_metrics.get('overall_pred_std_s', float('nan')):.6f}"
        f", experimental_pred_min_s={exp_metrics.get('pred_min_s', float('nan')):.4f}"
        f", experimental_pred_p05_s={exp_metrics.get('pred_p05_s', float('nan')):.4f}"
        f", experimental_pred_p25_s={exp_metrics.get('pred_p25_s', float('nan')):.4f}"
        f", experimental_pred_p50_s={exp_metrics.get('pred_p50_s', float('nan')):.4f}"
        f", experimental_pred_p75_s={exp_metrics.get('pred_p75_s', float('nan')):.4f}"
        f", experimental_pred_p95_s={exp_metrics.get('pred_p95_s', float('nan')):.4f}"
        f", experimental_pred_max_s={exp_metrics.get('pred_max_s', float('nan')):.4f}"
    )
    print(final_msg, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train exact-3 transformer in the legacy two-stage regime.")
    parser.add_argument("--meta-dir", default="data/dct_meta3")
    parser.add_argument("--runs-dir", default="data/runs3")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--monitor-dir", default=None)
    parser.add_argument("--tag", required=True, default='summary.json')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp-run-dir", default="data/experimental")
    parser.add_argument("--exp-onset-json", default="dct_onset_exp_dct_actual.json")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--inp-seq-len", type=int, default=1500)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-step-offset", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=6000)
    parser.add_argument("--min-epochs-before-validation", type=int, default=0)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-traj-train", "--train-pool-size", dest="n_traj_train", type=int, default=None)
    parser.add_argument("--n-traj-burst-val", "--val-size", dest="n_traj_burst_val", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--finetune-drop-to-min-sensors", type=int, default=None)
    parser.add_argument("--finetune-drop-to-max-sensors", type=int, default=None)
    parser.add_argument(
        "--amp-train",
        action="store_true",
        help="Train the transformer under CUDA BF16 autocast.",
    )
    parser.add_argument(
        "--no-amp-eval",
        action="store_true",
        help="Disable CUDA BF16 autocast during validation/test/experimental evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    is_finetune = args.init_checkpoint is not None
    cfg = Config(
        inp_seq_len=args.inp_seq_len,
        lr=args.lr if args.lr is not None else (0.08 if is_finetune else 0.6),
        warmup_step_offset=args.warmup_step_offset,
        max_epochs=args.max_epochs,
        min_epochs_before_validation=args.min_epochs_before_validation,
        min_epochs_before_early_stop=args.min_epochs_before_early_stop,
        patience=args.patience,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        runs_dir=args.runs_dir,
        seed=args.seed,
        use_amp_train=args.amp_train,
        use_amp_eval=not args.no_amp_eval,
        requested_test_size=args.test_size,
        **({"requested_val_size": args.n_traj_burst_val} if args.n_traj_burst_val is not None else {}),
        **({"n_traj_train": args.n_traj_train} if args.n_traj_train is not None else {}),
        **({"n_traj_burst_val": args.n_traj_burst_val} if args.n_traj_burst_val is not None else {}),
        **({"finetune_drop_to_min_sensors": args.finetune_drop_to_min_sensors} if args.finetune_drop_to_min_sensors is not None else {}),
        **({"finetune_drop_to_max_sensors": args.finetune_drop_to_max_sensors} if args.finetune_drop_to_max_sensors is not None else {}),
    )
    run_training(
        meta_dir=args.meta_dir,
        out_dir=args.out_dir,
        exp_run_dir=args.exp_run_dir,
        exp_onset_json=args.exp_onset_json,
        device_name=args.device,
        cfg=cfg,
        tag=args.tag,
        init_checkpoint=args.init_checkpoint,
        monitor_dir=args.monitor_dir,
    )


if __name__ == "__main__":
    main()
