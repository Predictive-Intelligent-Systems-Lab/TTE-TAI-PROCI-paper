from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.data as _tud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import lstm_reg_model
from preprocess_mod import WindowDataset
from rebuttal_analysis.common import DT, ensure_dir, load_metadata, set_seed, write_json
from trans_train import (
    compute_uniform_label_log_stats,
    compute_pool_sizes_like_original,
    evaluate_experiment,
    evaluate_loader,
    split_trajectory_ids,
    subset_metadata_by_ids,
    worker_init_fn,
)


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
    best_val_mae_s: float | None,
    latest_history: dict | None,
    history_length: int,
    training_in_progress: bool,
    test_metrics: dict | None = None,
    exp_metrics: dict | None = None,
) -> None:
    ensure_dir(monitor_dir)
    payload: dict = {
        "tag": tag,
        "state": state,
        "summary_path": os.path.abspath(summary_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_val_mae_s": best_val_mae_s,
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


def make_loader(dct_meta: dict, cfg, train: bool, pool_sizes: dict) -> _tud.DataLoader:
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
        dynamic_sensor_sampling=False,
        biased_sampling=False,
        train=train,
        i_sens_val=cfg.exact_n_sensors - 1,
        exact_n_sensors=cfg.exact_n_sensors,
        runs_dir=cfg.runs_dir,
    )
    return _tud.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.num_workers_train if train else cfg.num_workers_val,
        pin_memory=True,
        prefetch_factor=4 if train else 2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )


def make_subset_loader(
    dct_meta_subset: dict,
    cfg,
    *,
    train: bool,
    n_traj_burst: int,
) -> _tud.DataLoader:
    subset_pool_sizes = {
        "n_traj_train": int(len(dct_meta_subset)) if train else 0,
        "n_traj_burst_train": int(min(n_traj_burst, len(dct_meta_subset))) if train else 0,
        "n_traj_burst_val": int(min(n_traj_burst, len(dct_meta_subset))) if not train else 0,
    }
    return make_loader(dct_meta_subset, cfg, train=train, pool_sizes=subset_pool_sizes)


@dataclass
class LSTMConfig:
    inp_seq_len: int = 1500
    input_dim: int = 24
    proj_dim: int = 64
    hidden_dim: int = 192
    num_layers: int = 2
    regression_head: int = 192
    dropout: float = 0.07
    batch_size: int = 512
    lr: float = 3e-4
    beta2: float = 0.999
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    max_epochs: int = 6000
    min_epochs_before_validation: int = 0
    patience: int = 1500
    min_delta: float = 5e-4
    min_epochs_before_early_stop: int = 5000
    inp_buff: int = 10240
    num_workers_train: int = 4
    num_workers_val: int = 2
    samples_per_epoch_train: int = 8192
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
    seed: int = 1337
    runs_dir: str = "data/runs3"


def make_model(cfg: LSTMConfig, device: torch.device) -> lstm_reg_model:
    return lstm_reg_model(
        input_dim=cfg.input_dim,
        proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        regression_head=cfg.regression_head,
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


def evaluate_loader_one_to_three(
    model: torch.nn.Module,
    loader: _tud.DataLoader,
    device: torch.device,
    lab_mean_log: float,
    lab_std_log: float,
    cfg: LSTMConfig,
    *,
    seed_offset: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mae_z_sum = 0.0
    mae_s_sum = 0.0
    pred_std_sum = 0.0
    targ_std_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (x, sens_ind_msk, labels_log) in enumerate(loader):
            rng = np.random.default_rng(cfg.seed + seed_offset + batch_idx)
            sens_ind_msk = sample_sensor_subset(
                sens_ind_msk,
                rng=rng,
                min_active=cfg.finetune_drop_to_min_sensors,
                max_active=cfg.finetune_drop_to_max_sensors,
            )
            x = x * sens_ind_msk[:, None, :].to(dtype=x.dtype)
            x = torch.concat([x, sens_ind_msk[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            labels_log = labels_log.to(device, dtype=torch.float32, non_blocking=True)
            labels_z = (labels_log - lab_mean_log) / lab_std_log

            pred_z = model(x)
            loss = model.loss_fun(pred_z, labels_z)
            pred_log = pred_z.float() * lab_std_log + lab_mean_log
            pred_s = torch.exp(pred_log)
            true_s = torch.exp(labels_log)

            loss_sum += float(loss.item())
            mae_z_sum += float(torch.abs(pred_z.float() - labels_z).mean().item())
            mae_s_sum += float(torch.abs(pred_s - true_s).mean().item() * DT)
            pred_std_sum += float(pred_z.float().std().item())
            targ_std_sum += float(labels_z.float().std().item())
            n_batches += 1

    return {
        "loss": loss_sum / max(n_batches, 1),
        "mae_z": mae_z_sum / max(n_batches, 1),
        "mae_s": mae_s_sum / max(n_batches, 1),
        "pred_std_val": pred_std_sum / max(n_batches, 1),
        "targ_std_val": targ_std_sum / max(n_batches, 1),
    }


def run_training(
    meta_dir: str,
    out_dir: str,
    exp_run_dir: str,
    exp_onset_json: str,
    device_name: str,
    cfg: LSTMConfig,
    tag: str,
    monitor_dir: str | None,
) -> None:
    ensure_dir(out_dir)
    monitor_dir = resolve_monitor_dir(out_dir, monitor_dir)
    ensure_dir(monitor_dir)
    set_seed(cfg.seed)
    device = torch.device(device_name)

    dct_meta, _ = load_metadata(meta_dir)
    cfg_effective = cfg
    split_ids = split_trajectory_ids(dct_meta, cfg_effective)
    pool_sizes = compute_pool_sizes_like_original(dct_meta, cfg_effective)
    train_meta = subset_metadata_by_ids(dct_meta, split_ids["train_ids"])
    val_meta = subset_metadata_by_ids(dct_meta, split_ids["val_ids"])
    test_meta = subset_metadata_by_ids(dct_meta, split_ids["test_ids"])
    lab_mean_log, lab_std_log = compute_uniform_label_log_stats(cfg.inp_buff)

    train_loader = make_subset_loader(train_meta, cfg_effective, train=True, n_traj_burst=pool_sizes["n_traj_burst_train"])
    val_loader = make_subset_loader(val_meta, cfg_effective, train=False, n_traj_burst=pool_sizes["n_traj_burst_val"])
    test_loader = None
    if pool_sizes["n_traj_burst_test"] > 0:
        test_loader = make_subset_loader(test_meta, cfg_effective, train=False, n_traj_burst=pool_sizes["n_traj_burst_test"])

    model = make_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, cfg.beta2),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=40,
        min_lr=1e-6,
    )

    best_val_mae_z = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_ckpt = os.path.join(out_dir, f"{tag}_best.pth")
    summary_path = os.path.join(out_dir, f"{tag}_summary.json")
    history = []

    write_json(
        summary_path,
        {
            "tag": tag,
            "meta_dir": os.path.abspath(meta_dir),
            "device": str(device),
            "best_epoch": int(best_epoch),
            "best_val_mae_s": None,
            "train_config": asdict(cfg),
            "pool_sizes": pool_sizes,
            "label_log_mean": float(lab_mean_log),
            "label_log_std": float(lab_std_log),
            "test_metrics": None,
            "experimental_transfer": {},
            "history": history,
            "checkpoint_path": os.path.abspath(best_ckpt),
            "training_in_progress": True,
        },
    )
    write_training_monitor(
        monitor_dir,
        tag,
        state="starting",
        summary_path=summary_path,
        checkpoint_path=best_ckpt,
        best_epoch=best_epoch,
        best_val_mae_s=None,
        latest_history=None,
        history_length=0,
        training_in_progress=True,
        test_metrics=None,
    )

    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss = 0.0
        total_mae_z = 0.0
        n_batches = 0
        pred_std_tr_sum = 0.0
        targ_std_tr_sum = 0.0

        for batch_idx, (x, sens_ind_msk, labels_log) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            rng = np.random.default_rng(cfg.seed + 100000 * epoch + batch_idx)
            sens_ind_msk = sample_sensor_subset(
                sens_ind_msk,
                rng=rng,
                min_active=cfg.finetune_drop_to_min_sensors,
                max_active=cfg.finetune_drop_to_max_sensors,
            )
            x = x * sens_ind_msk[:, None, :].to(dtype=x.dtype)
            x = torch.concat([x, sens_ind_msk[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            labels_log = labels_log.to(device, dtype=torch.float32, non_blocking=True)
            labels_z = (labels_log - lab_mean_log) / lab_std_log

            pred_z = model(x)
            loss = model.loss_fun(pred_z, labels_z)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            total_mae_z += torch.abs(pred_z.detach() - labels_z).mean().item()
            n_batches += 1
            pred_std_tr_sum += pred_z.detach().float().std().item()
            targ_std_tr_sum += labels_z.detach().float().std().item()

        train_metrics = {
            "loss": total_loss / max(n_batches, 1),
            "mae_z": total_mae_z / max(n_batches, 1),
            "pred_std_tr": pred_std_tr_sum / max(n_batches, 1),
            "targ_std_tr": targ_std_tr_sum / max(n_batches, 1),
        }
        validation_active = epoch >= cfg.min_epochs_before_validation
        if validation_active:
            val_metrics = evaluate_loader(cfg, model, val_loader, device, lab_mean_log, lab_std_log)
            scheduler.step(val_metrics["mae_z"])
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

            if val_metrics["mae_z"] < best_val_mae_z - cfg.min_delta:
                best_val_mae_z = val_metrics["mae_z"]
                best_epoch = epoch
                patience_counter = 0
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
                )
                print(
                    f"[{tag}] new best checkpoint at epoch {epoch}: "
                    f"val_mae_z={best_val_mae_z:.6f} "
                    f"experimental_mae_s={exp_metrics_best.get('overall_mae_s', float('nan')):.6f} "
                    f"experimental_pred_std_s={exp_metrics_best.get('overall_pred_std_s', float('nan')):.6f} "
                    f"experimental_pred_p95_s={exp_metrics_best.get('pred_p95_s', float('nan')):.4f} "
                    f"experimental_pred_max_s={exp_metrics_best.get('pred_max_s', float('nan')):.4f}",
                    flush=True,
                )
            else:
                patience_counter += 1
                if epoch >= cfg.min_epochs_before_early_stop and patience_counter >= cfg.patience:
                    print(f"[{tag}] early stopping at epoch {epoch}", flush=True)
                    write_json(
                        summary_path,
                        {
                            "tag": tag,
                            "meta_dir": os.path.abspath(meta_dir),
                            "device": str(device),
                            "best_epoch": int(best_epoch),
                            "best_val_mae_z": float(best_val_mae_z) if best_val_mae_z < float("inf") else None,
                            "train_config": asdict(cfg),
                            "pool_sizes": pool_sizes,
                            "label_log_mean": float(lab_mean_log),
                            "label_log_std": float(lab_std_log),
                            "test_metrics": None,
                            "experimental_transfer": {},
                            "history": history,
                            "checkpoint_path": os.path.abspath(best_ckpt),
                            "training_in_progress": True,
                        },
                    )
                    write_training_monitor(
                        monitor_dir,
                        tag,
                        state="running",
                        summary_path=summary_path,
                        checkpoint_path=best_ckpt,
                        best_epoch=best_epoch,
                        best_val_mae_s=float(best_val_mae_z) if best_val_mae_z < float("inf") else None,
                        latest_history=history[-1] if history else None,
                        history_length=len(history),
                        training_in_progress=True,
                        test_metrics=None,
                    )
                    break
        else:
            history.append(
                {
                    "epoch": epoch,
                    "validation_active": False,
                    **train_metrics,
                }
            )
            print(
                f"[{tag}] epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_mae_z={train_metrics['mae_z']:.6f} "
                f"pred_std_tr={train_metrics['pred_std_tr']:.4f} "
                f"targ_std_tr={train_metrics['targ_std_tr']:.4f} "
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
                "best_epoch": int(best_epoch),
                "best_val_mae_z": float(best_val_mae_z) if best_val_mae_z < float("inf") else None,
                "train_config": asdict(cfg),
                "pool_sizes": pool_sizes,
                "label_log_mean": float(lab_mean_log),
                "label_log_std": float(lab_std_log),
                "test_metrics": None,
                "experimental_transfer": {},
                "history": history,
                "checkpoint_path": os.path.abspath(best_ckpt),
                "training_in_progress": True,
            },
        )
        write_training_monitor(
            monitor_dir,
            tag,
            state="running",
            summary_path=summary_path,
            checkpoint_path=best_ckpt,
            best_epoch=best_epoch,
            best_val_mae_s=float(best_val_mae_z) if best_val_mae_z < float("inf") else None,
            latest_history=history[-1] if history else None,
            history_length=len(history),
            training_in_progress=True,
            test_metrics=None,
        )

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
    model.load_state_dict(saved["model_state_dict"])
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate_loader(cfg, model, test_loader, device, lab_mean_log, lab_std_log)
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
    )

    write_json(
        summary_path,
        {
            "tag": tag,
            "meta_dir": os.path.abspath(meta_dir),
            "device": str(device),
            "best_epoch": int(best_epoch),
            "best_val_mae_z": float(best_val_mae_z),
            "train_config": asdict(cfg),
            "pool_sizes": pool_sizes,
            "label_log_mean": float(lab_mean_log),
            "label_log_std": float(lab_std_log),
            "test_metrics": test_metrics,
            "experimental_transfer": exp_metrics,
            "history": history,
            "checkpoint_path": os.path.abspath(best_ckpt),
            "training_in_progress": False,
        },
    )
    write_training_monitor(
        monitor_dir,
        tag,
        state="finished",
        summary_path=summary_path,
        checkpoint_path=best_ckpt,
        best_epoch=best_epoch,
        best_val_mae_s=float(best_val_mae_z) if best_val_mae_z < float("inf") else None,
        latest_history=history[-1] if history else None,
        history_length=len(history),
        training_in_progress=False,
        test_metrics=test_metrics,
        exp_metrics=exp_metrics,
    )
    final_msg = (
        f"[{tag}] finished: best_epoch={best_epoch}, "
        f"best_val_mae_z={(best_val_mae_z if best_val_mae_z < float('inf') else float('nan')):.6f}"
    )
    if test_metrics is not None:
        final_msg += f", test_mae_s={test_metrics['mae_s']:.6f}"
    final_msg += (
        f", experimental_mae_s={exp_metrics['overall_mae_s']:.6f}"
        f", experimental_pred_std_s={exp_metrics.get('overall_pred_std_s', float('nan')):.6f}"
        f", experimental_pred_p95_s={exp_metrics.get('pred_p95_s', float('nan')):.4f}"
        f", experimental_pred_max_s={exp_metrics.get('pred_max_s', float('nan')):.4f}"
    )
    print(final_msg, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM baseline seed study run.")
    parser.add_argument("--meta-dir", default="data/dct_meta3")
    parser.add_argument("--runs-dir", default="data/runs3")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--monitor-dir", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp-run-dir", default="data/experimental")
    parser.add_argument("--exp-onset-json", default="dct_onset_exp_dct_actual.json")
    parser.add_argument("--max-epochs", type=int, default=6000)
    parser.add_argument("--min-epochs-before-validation", type=int, default=0)
    parser.add_argument("--patience", type=int, default=1500)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=5000)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-traj-train", "--train-pool-size", dest="n_traj_train", type=int, default=None)
    parser.add_argument("--n-traj-burst-val", "--val-size", dest="n_traj_burst_val", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_traj_train = args.n_traj_train if args.n_traj_train is not None else 900
    n_traj_burst_val = args.n_traj_burst_val if args.n_traj_burst_val is not None else 259
    cfg = LSTMConfig(
        max_epochs=args.max_epochs,
        min_epochs_before_validation=args.min_epochs_before_validation,
        patience=args.patience,
        min_epochs_before_early_stop=args.min_epochs_before_early_stop,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        seed=args.seed,
        runs_dir=args.runs_dir,
        n_traj_train=n_traj_train,
        n_traj_burst_val=n_traj_burst_val,
        requested_val_size=n_traj_burst_val,
        requested_test_size=args.test_size,
        **({"lr": args.lr} if args.lr is not None else {}),
    )
    run_training(
        meta_dir=args.meta_dir,
        out_dir=args.out_dir,
        exp_run_dir=args.exp_run_dir,
        exp_onset_json=args.exp_onset_json,
        device_name=args.device,
        cfg=cfg,
        tag=args.tag,
        monitor_dir=args.monitor_dir,
    )


if __name__ == "__main__":
    main()
