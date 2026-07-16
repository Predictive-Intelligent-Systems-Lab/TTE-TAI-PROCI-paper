# train_xgb_env_precursors_3sens.py
#
# XGBoost baseline on FOUR handcrafted envelope precursors:
#   - var_env     : variance of |P1(t)|
#   - ar1_env     : lag-L autocorrelation of |P1(t)|
#   - kurt_env    : kurtosis of |P1(t)|
#   - mode1_frac  : projection-based mode-1 energy fraction proxy
#
# Corrected for 3-sensor experimental matching:
#   - synthetic training uses only 3 selected sensors
#   - mode-1 reconstruction uses the actual 3 sensor angles
#   - same observation operator should be reused on experiment
#
# Target:
#   log(time-to-onset in samples), same as transformer
#
# Run:
#   python train_xgb_env_precursors_3sens.py

import os
import re
import math
import json
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, welch, decimate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from rebuttal_analysis.common import CURRENT_RELABEL_LOGIC, SyntheticOnsetConfig
from rebuttal_analysis.relabel_synthetic_data import relabel_directory


# =========================
# Reproducibility
# =========================
SEED = 1343


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Config
# =========================
@dataclass
class Config:
    # file locations
    run_dir: str = "data/runs3"
    meta_dir: str = "data/dct_meta3"
    run_prefix: str = "sim_"
    run_suffix: str = ".pt"
    meta_suffix: str = ".pth"

    # raw sampling
    dt_raw: float = 1.953125e-04   # 5120 Hz synthetic data
    n_sensors_total: int = 12

    # choose the 3 synthetic sensor channels that best match the experimental azimuths
    sensor_indices_in_synth = tuple([0, 4, 8])

    # actual azimuths of the 3 experimental sensors
    sensor_angles_rad= np.linspace(0, 2*np.pi, 3, endpoint=False)
    #     0.0,
    #     2.0 * math.pi / 3.0,
    #     4.0 * math.pi / 3.0,
    # )

    # split
    n_traj_train: int = 900
    split_seed: int = 0

    # window sampling, aligned with transformer code
    inp_buff: int = 10240
    biased_sampling: bool = False
    inter_samp_space: int = 1
    train_windows_per_traj: int = 64
    val_windows_per_traj: int = 64

    # feature extraction
    feature_win_s: float = 1.0      # same as WIN_S in plot script
    f0_est_win_s: float = 1.0
    fs_env: float = 160.0
    f_env_lp: float = 60.0
    f0_min: float = 200.0
    f0_max: float = 2000.0

    # precursor params from plot script
    ar_lag_L: int = 5
    mode1_frac_mmax: int = 2

    # XGBoost
    xgb_n_estimators: int = 1200
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 4
    xgb_min_child_weight: int = 8
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 2.0
    xgb_early_stopping_rounds: int = 50
    xgb_tree_method: str = "hist"

    # outputs
    out_dir: str = f"xgb_env_precursors_norminput_3sens_{SEED}"

    # onset relabelling
    relabel_training_onsets: bool = True
    relabel_alpha: float = 0.03
    relabel_persistence_samples: int = 4000
    relabel_persistence_fraction: float = 0.15
    relabel_pers_adjust: int = 1000
    relabel_workers: int = 4
    relabel_meta_dir: str = (
        "rebuttal_analysis/results/synthetic_meta/"
        "xgboost_runs3_thresh_a0p03_pq4000_pa1000"
    )


CFG = Config()
FS_RAW = 1.0 / CFG.dt_raw
DECIM = int(round(FS_RAW / CFG.fs_env))
assert abs(FS_RAW / DECIM - CFG.fs_env) < 1e-9, "fs_env must divide fs_raw exactly"

SENSOR_IDXS = np.array(CFG.sensor_indices_in_synth, dtype=int)
SENSOR_ANGLES = np.array(CFG.sensor_angles_rad, dtype=np.float64)

FEATURE_COLUMNS = [
    "var_env_mode1_3sens",
    "ar1_env_mode1_3sens",
    "kurt_env_mode1_3sens",
    "mode1_frac_3sens",
]

# if len(SENSOR_IDXS) != 3:
#     raise ValueError("This script expects exactly 3 synthetic sensor indices")
# if len(SENSOR_ANGLES) != 3:
#     raise ValueError("This script expects exactly 3 sensor angles")
if np.any(SENSOR_IDXS < 0) or np.any(SENSOR_IDXS >= CFG.n_sensors_total):
    raise ValueError("sensor_indices_in_synth contains invalid channel indices")


# =========================
# Utilities
# =========================
def sim_id_from_name(path: str) -> int:
    m = re.search(r"sim_(\d+)", os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse sim id from {path}")
    return int(m.group(1))


def load_meta_dict(meta_dir: str) -> dict[int, dict]:
    dct_meta = {}
    for fname in os.listdir(meta_dir):
        full = os.path.join(meta_dir, fname)
        if not os.path.isfile(full):
            continue
        if not fname.startswith(CFG.run_prefix) or not fname.endswith(CFG.meta_suffix):
            continue
        sid = sim_id_from_name(fname)
        dct_meta[sid] = torch.load(full, weights_only=False)
    return dct_meta


def build_relabel_cfg() -> SyntheticOnsetConfig:
    return SyntheticOnsetConfig(
        alpha=CFG.relabel_alpha,
        persistence_samples=CFG.relabel_persistence_samples,
        persistence_fraction=CFG.relabel_persistence_fraction,
        pers_adjust=CFG.relabel_pers_adjust,
    )


def meta_dir_matches_relabel_cfg(meta_dir: str, run_dir: str, src_meta_dir: str, cfg: SyntheticOnsetConfig) -> bool:
    summary_path = os.path.join(meta_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return False

    if summary.get("label_logic") != CURRENT_RELABEL_LOGIC:
        return False
    if float(summary.get("alpha", float("nan"))) != float(cfg.alpha):
        return False
    if int(summary.get("persistence_samples", -1)) != int(cfg.persistence_samples):
        return False
    if float(summary.get("persistence_fraction", float("nan"))) != float(cfg.persistence_fraction):
        return False
    if int(summary.get("pers_adjust", -1)) != int(cfg.pers_adjust):
        return False
    if os.path.abspath(summary.get("runs_dir", "")) != os.path.abspath(run_dir):
        return False
    if os.path.abspath(summary.get("src_meta_dir", "")) != os.path.abspath(src_meta_dir):
        return False

    return any(
        name.endswith(".pth") and os.path.isfile(os.path.join(meta_dir, name))
        for name in os.listdir(meta_dir)
    )


def resolve_training_meta_dir() -> tuple[str, dict[str, object]]:
    if not CFG.relabel_training_onsets:
        return CFG.meta_dir, {"label_source": "meta_dir"}

    relabel_cfg = build_relabel_cfg()
    if meta_dir_matches_relabel_cfg(CFG.meta_dir, CFG.run_dir, CFG.meta_dir, relabel_cfg):
        return CFG.meta_dir, {
            "label_source": "meta_dir",
            "label_logic": CURRENT_RELABEL_LOGIC,
            "relabel_cfg": asdict(relabel_cfg),
        }

    if not meta_dir_matches_relabel_cfg(CFG.relabel_meta_dir, CFG.run_dir, CFG.meta_dir, relabel_cfg):
        print(
            "[labels] relabelling synthetic onsets from common.py "
            f"into {CFG.relabel_meta_dir}",
            flush=True,
        )
        relabel_directory(
            runs_dir=CFG.run_dir,
            src_meta_dir=CFG.meta_dir,
            dst_meta_dir=CFG.relabel_meta_dir,
            cfg=relabel_cfg,
            n_workers=CFG.relabel_workers,
        )

    return CFG.relabel_meta_dir, {
        "label_source": "relabelled_meta_dir",
        "label_logic": CURRENT_RELABEL_LOGIC,
        "relabel_cfg": asdict(relabel_cfg),
    }


def make_train_val_split(all_ids: np.ndarray, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_ids = rng.choice(all_ids, size=n_train, replace=False)
    val_ids = np.setdiff1d(all_ids, train_ids, assume_unique=True)
    return np.sort(train_ids), np.sort(val_ids)


def get_sampling_distribution(inp_buff: int, biased_sampling: bool, inter_samp_space: int = 1) -> tuple[np.ndarray, np.ndarray]:
    sampling_space = np.arange(1, inp_buff + 1, inter_samp_space)
    if biased_sampling:
        probs = np.linspace(1.0, 0.0, sampling_space.shape[0], dtype=np.float64)
    else:
        probs = np.ones_like(sampling_space, dtype=np.float64)
    probs /= probs.sum()
    return sampling_space, probs


def butter_lowpass_complex(x: np.ndarray, fs: float, f_cut: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, f_cut / (fs / 2.0), btype="low")
    xr = filtfilt(b, a, np.real(x), axis=0)
    xi = filtfilt(b, a, np.imag(x), axis=0)
    return xr + 1j * xi


def estimate_carrier_f0(seg: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """
    seg: [T, 3] real-valued pressure segment
    """
    x = np.mean(seg, axis=1)
    nper = min(4096, len(x))
    if nper < 256:
        return 750.0
    f, Pxx = welch(x, fs=fs, nperseg=nper)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return 750.0
    return float(f[m][np.argmax(Pxx[m])])


def demodulate_envelope(seg: np.ndarray, fs_raw: float, f0: float, f_env_lp: float, decim: int) -> np.ndarray:
    """
    seg: [T_raw, 3] real
    returns z_env: [T_env, 3] complex envelope
    """
    t = np.arange(seg.shape[0]) / fs_raw
    mix = np.exp(-1j * 2.0 * np.pi * f0 * t)[:, None]
    z = seg * mix
    z_lp = butter_lowpass_complex(z, fs_raw, f_env_lp, order=4)

    zr = decimate(np.real(z_lp), decim, ftype="fir", zero_phase=True, axis=0)
    zi = decimate(np.imag(z_lp), decim, ftype="fir", zero_phase=True, axis=0)
    return zr + 1j * zi


def project_mode1_from_3sensors(z_env: np.ndarray, sensor_angles: np.ndarray) -> np.ndarray:
    """
    Reconstruct first azimuthal mode from 3 complex sensor envelopes using
    least-squares projection onto cos(theta), sin(theta):

        z(theta, t) ≈ a(t) cos(theta) + b(t) sin(theta)

    Then build complex mode-1 signal:
        P1(t) = a(t) - i b(t)

    z_env: [T, 3] complex
    sensor_angles: [3]
    returns: [T] complex mode-1 amplitude proxy
    """
    # if z_env.shape[1] != 3:
    #     raise ValueError(f"Expected 3 sensors, got {z_env.shape[1]}")

    H = np.column_stack([
        np.cos(sensor_angles),
        np.sin(sensor_angles),
    ])  # [3, 2]

    H_pinv = np.linalg.pinv(H)      # [2, 3]
    coeffs = z_env @ H_pinv.T       # [T, 2], complex
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    P1 = a - 1j * b
    return P1


def ar_lag_coeff(x: np.ndarray, L: int) -> float | None:
    if len(x) <= L:
        return None
    x = x.astype(np.float64)
    x = x - x.mean()
    x0 = x[:-L]
    x1 = x[L:]
    denom = np.dot(x0, x0) + 1e-12
    return float(np.dot(x0, x1) / denom)


def kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    v = np.var(x) + 1e-12
    return float(((x - x.mean()) ** 4).mean() / (v * v))


def mode1_fraction_from_3sensors(
    z_env: np.ndarray,
    sensor_angles: np.ndarray,
    m_max: int = 2,
) -> float:
    """
    For 3 uniformly spaced sensors, compute the resolved non-axisymmetric
    mode-1 energy fraction:

        E1 / (E1 + E2)

    where E1 and E2 are the energies of the two nonzero spatial bins of the
    3-point azimuthal decomposition.

    Note:
      - m=3 is not separately observable with 3 sensors; it aliases into m=0.
      - therefore this is NOT E1 / (E1 + E2 + E3).
    """
    # if z_env.ndim != 2 or z_env.shape[1] != 3:
    #     raise ValueError(f"Expected z_env shape [T, 3], got {z_env.shape}")

    angles = np.asarray(sensor_angles, dtype=np.float64)

    # Optional sanity check for uniform spacing
    # diffs = np.sort((angles - angles[0]) % (2 * np.pi))
    # diffs = np.diff(np.r_[diffs, 2 * np.pi])
    # if not np.allclose(np.sort(diffs), np.full(3, 2 * np.pi / 3), atol=1e-6):
    #     raise ValueError("sensor_angles are not uniformly spaced by 2π/3")

    energies = []
    for m in range(m_max + 1):
        w = np.exp(-1j * m * angles)[None, :]
        Pm = np.mean(z_env * w, axis=1)
        energies.append(np.mean(np.abs(Pm) ** 2))

    E = np.asarray(energies, dtype=np.float64)
    return float(E[1] / (np.sum(E[1:]) + 1e-12))


def extract_env_precursor_features(
    traj_12: np.ndarray,
    k_end: int,
    onset: int,
    inp_buff: int,
    feature_win_raw: int,
    f0_est_win_raw: int,
    sensor_indices: np.ndarray,
    sensor_angles: np.ndarray,
) -> dict[str, float] | None:
    """
    Causal handcrafted features on the last feature_win_s before k_end,
    reconstructed from 3 selected sensors only.

    Input is normalized per-sensor using the reference segment
    traj[onset - inp_buff - feature_win_raw : k_end], matching the
    sequence-model convention.

    Features:
      - variance of |P1(t)|
      - lag-L autocorrelation of |P1(t)|
      - kurtosis of |P1(t)|
      - projection-based mode1_fraction proxy from z_env
    """
    if traj_12.ndim != 2 or traj_12.shape[1] != CFG.n_sensors_total:
        raise ValueError(f"Expected full synthetic trajectory shape [T, 12], got {traj_12.shape}")

    if k_end <= max(feature_win_raw, f0_est_win_raw):
        return None

    traj = traj_12[:, sensor_indices]  # [T, 3]

    # Per-sensor normalisation using the historical context
    ref_start = max(0, onset - inp_buff - feature_win_raw)
    ref = traj[ref_start : k_end, :]          # [T_ref, 3]
    if len(ref) < 2:
        return None
    ref_mean = ref.mean(axis=0)               # [3]
    ref_std  = ref.std(axis=0)                # [3]
    if np.any(ref_std < 1e-12):
        return None

    seg_feat = (traj[k_end - feature_win_raw : k_end, :] - ref_mean) / ref_std
    seg_f0   = traj[k_end - f0_est_win_raw   : k_end, :]  # f0 estimation is amplitude-invariant

    f0 = estimate_carrier_f0(seg_f0, FS_RAW, CFG.f0_min, CFG.f0_max)
    z_env = demodulate_envelope(seg_feat, FS_RAW, f0, CFG.f_env_lp, DECIM)
    if z_env.shape[0] <= max(8, CFG.ar_lag_L):
        return None

    P1 = project_mode1_from_3sensors(z_env, sensor_angles)
    amp = np.abs(P1)

    ar1 = ar_lag_coeff(amp, CFG.ar_lag_L)
    if ar1 is None:
        return None

    feats = {
        "var_env_mode1_3sens": float(np.var(amp)),
        "ar1_env_mode1_3sens": float(ar1),
        "kurt_env_mode1_3sens": float(kurtosis(amp)),
        "mode1_frac_3sens": float(
            mode1_fraction_from_3sensors(
                z_env=z_env,
                sensor_angles=sensor_angles,
                m_max=CFG.mode1_frac_mmax,
            )
        ),
    }

    if not all(np.isfinite(v) for v in feats.values()):
        return None

    return feats


def build_feature_table(
    traj_ids: np.ndarray,
    dct_meta: dict[int, dict],
    windows_per_traj: int,
    sampling_space: np.ndarray,
    sampling_probs: np.ndarray,
    feature_win_raw: int,
    f0_est_win_raw: int,
    verbose_every: int = 25,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)

    n_ok = 0
    n_skip_short = 0
    n_skip_badfeat = 0

    for i, tid in enumerate(traj_ids):
        meta = dct_meta[int(tid)]
        onset = int(meta["onset_ts"])

        min_needed = max(CFG.inp_buff + feature_win_raw, feature_win_raw, f0_est_win_raw)
        if onset <= min_needed:
            n_skip_short += 1
            continue

        run_path = os.path.join(CFG.run_dir, f"sim_{tid}.pt")
        traj = torch.load(run_path, mmap=True).numpy()

        for _ in range(windows_per_traj):
            dist_to_onset = int(rng.choice(sampling_space, p=sampling_probs))
            k_end = onset - dist_to_onset

            feats = extract_env_precursor_features(
                traj_12=traj,
                k_end=k_end,
                onset=onset,
                inp_buff=CFG.inp_buff,
                feature_win_raw=feature_win_raw,
                f0_est_win_raw=f0_est_win_raw,
                sensor_indices=SENSOR_IDXS,
                sensor_angles=SENSOR_ANGLES,
            )

            if feats is None:
                n_skip_badfeat += 1
                continue

            row = {
                "sid": int(tid),
                "k_end": int(k_end),
                "dist_to_onset": int(dist_to_onset),
                "log_dist_to_onset": float(math.log(dist_to_onset)),
            }
            row.update(feats)
            rows.append(row)

        n_ok += 1
        if (i + 1) % verbose_every == 0:
            print(
                f"{i+1}/{len(traj_ids)} trajectories processed | "
                f"usable={n_ok}, short={n_skip_short}, badfeat={n_skip_badfeat}, "
                f"rows={len(rows)}"
            )

    df = pd.DataFrame(rows)
    print(
        f"Done. trajectories={len(traj_ids)} | usable={n_ok} | "
        f"short={n_skip_short} | badfeat={n_skip_badfeat} | rows={len(df)}"
    )
    return df


def eval_regression(y_true_log: np.ndarray, y_pred_log: np.ndarray, fs_raw: float) -> dict[str, float]:
    out = {}

    out["mae_log"] = float(mean_absolute_error(y_true_log, y_pred_log))
    out["mse_log"] = float(mean_squared_error(y_true_log, y_pred_log))
    out["rmse_log"] = float(np.sqrt(out["mse_log"]))

    true_t_s = np.exp(y_true_log) / fs_raw
    pred_t_s = np.exp(y_pred_log) / fs_raw
    out["mae_seconds"] = float(mean_absolute_error(true_t_s, pred_t_s))
    out["mse_seconds"] = float(mean_squared_error(true_t_s, pred_t_s))
    out["rmse_seconds"] = float(np.sqrt(out["mse_seconds"]))

    return out


def main() -> None:
    set_seed(SEED)
    os.makedirs(CFG.out_dir, exist_ok=True)

    feature_win_raw = int(round(CFG.feature_win_s * FS_RAW))
    f0_est_win_raw = int(round(CFG.f0_est_win_s * FS_RAW))

    sampling_space, sampling_probs = get_sampling_distribution(
        inp_buff=CFG.inp_buff,
        biased_sampling=CFG.biased_sampling,
        inter_samp_space=CFG.inter_samp_space,
    )

    meta_dir, label_info = resolve_training_meta_dir()
    dct_meta = load_meta_dict(meta_dir)
    all_ids = np.array(sorted(dct_meta.keys()), dtype=np.int64)
    if CFG.n_traj_train > len(all_ids):
        raise ValueError(f"n_traj_train={CFG.n_traj_train} > total trajectories={len(all_ids)}")

    train_ids, val_ids = make_train_val_split(all_ids, CFG.n_traj_train, CFG.split_seed)
    print(f"Total trajectories: {len(all_ids)}")
    print(f"Train trajectories: {len(train_ids)}")
    print(f"Val trajectories:   {len(val_ids)}")
    print(f"Training meta dir:          {meta_dir}")
    print(f"Training label source:      {label_info['label_source']}")
    print(f"Synthetic sensor idxs used: {SENSOR_IDXS.tolist()}")
    print(f"Sensor angles used (rad):   {SENSOR_ANGLES.tolist()}")
    print(f"Feature columns:            {FEATURE_COLUMNS}")

    print("\nBuilding TRAIN table...")
    df_train = build_feature_table(
        traj_ids=train_ids,
        dct_meta=dct_meta,
        windows_per_traj=CFG.train_windows_per_traj,
        sampling_space=sampling_space,
        sampling_probs=sampling_probs,
        feature_win_raw=feature_win_raw,
        f0_est_win_raw=f0_est_win_raw,
    )

    print("\nBuilding VAL table...")
    df_val = build_feature_table(
        traj_ids=val_ids,
        dct_meta=dct_meta,
        windows_per_traj=CFG.val_windows_per_traj,
        sampling_space=sampling_space,
        sampling_probs=sampling_probs,
        feature_win_raw=feature_win_raw,
        f0_est_win_raw=f0_est_win_raw,
    )

    if len(df_train) == 0 or len(df_val) == 0:
        raise RuntimeError("Empty train or val table. Check paths and onset validity.")

    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = df_train["log_dist_to_onset"].values.astype(np.float32)

    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = df_val["log_dist_to_onset"].values.astype(np.float32)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=CFG.xgb_n_estimators,
        learning_rate=CFG.xgb_learning_rate,
        max_depth=CFG.xgb_max_depth,
        min_child_weight=CFG.xgb_min_child_weight,
        subsample=CFG.xgb_subsample,
        colsample_bytree=CFG.xgb_colsample_bytree,
        reg_alpha=CFG.xgb_reg_alpha,
        reg_lambda=CFG.xgb_reg_lambda,
        random_state=SEED,
        tree_method=CFG.xgb_tree_method,
        n_jobs=-1,
        early_stopping_rounds=CFG.xgb_early_stopping_rounds,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)

    metrics_train = eval_regression(y_train, yhat_train, FS_RAW)
    metrics_val = eval_regression(y_val, yhat_val, FS_RAW)

    print("\nTRAIN metrics")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.6f}")

    print("\nVAL metrics")
    for k, v in metrics_val.items():
        print(f"{k}: {v:.6f}")

    model_path = os.path.join(CFG.out_dir, "xgb_env_precursors_3sens_model.json")
    model.save_model(model_path)

    df_train.to_csv(os.path.join(CFG.out_dir, "train_features_env_precursors_3sens.csv"), index=False)
    df_val.to_csv(os.path.join(CFG.out_dir, "val_features_env_precursors_3sens.csv"), index=False)

    with open(os.path.join(CFG.out_dir, "metrics_train.json"), "w") as f:
        json.dump(metrics_train, f, indent=2)

    with open(os.path.join(CFG.out_dir, "metrics_val.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)

    config_payload = asdict(CFG)
    config_payload["effective_meta_dir"] = meta_dir
    config_payload["training_label_info"] = label_info
    with open(os.path.join(CFG.out_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=2)

    with open(os.path.join(CFG.out_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()
