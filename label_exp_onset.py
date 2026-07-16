"""
Label experimental ramps with the same quantile-threshold persistence onset
finder used for synthetic relabelling.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPT_PARENT = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

if SCRIPT_PARENT == "rebuttal_analysis":
    from common import (
        SyntheticOnsetConfig,
        load_experimental_run,
        relabel_onset_from_saved_run,
    )


def find_onset(
    p_sens,
    run_id: int,
    cfg: SyntheticOnsetConfig,
    *,
    verbose: bool = True,
) -> int | None:
    onset = relabel_onset_from_saved_run(p_sens, cfg)
    if onset is None:
        if verbose:
            print(f"  run {run_id}: no onset found")
        return None
    if verbose:
        print(f"  run {run_id}: onset={onset}")
    return int(onset)


def _label_one_worker(args: tuple[int, str, SyntheticOnsetConfig]) -> tuple[int, int | None]:
    run_id, exp_dir, cfg = args
    path = os.path.join(exp_dir, f"ramp_{run_id}.pt")
    traj = load_experimental_run(path).numpy()
    onset = find_onset(traj, run_id, cfg, verbose=False)
    return run_id, onset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relabel experimental onset JSON with the quantile-threshold persistence onset finder.")
    parser.add_argument("--exp-dir", default="data/experimental")
    parser.add_argument("--out-json", default="dct_onset_exp_settle.json")
    parser.add_argument("--onset-thresh", "--alpha", dest="alpha", type=float, default=0.03)
    parser.add_argument(
        "--pers-quantile",
        "--persistence-samples",
        "--persistence-window",
        dest="persistence_samples",
        type=int,
        default=4000,
    )
    parser.add_argument("--pers-adjust", type=int, default=2000)
    parser.add_argument("--persistence-fraction", type=float, default=0.15)

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes. Set to -1 to use all CPU cores.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = SyntheticOnsetConfig(
        alpha=float(args.alpha),
        persistence_samples=int(args.persistence_samples),
        persistence_fraction=float(args.persistence_fraction),
        pers_adjust=int(args.pers_adjust),
    )

    run_ids = sorted(
        int(f.replace("ramp_", "").replace(".pt", ""))
        for f in os.listdir(args.exp_dir)
        if f.startswith("ramp_") and f.endswith(".pt")
    )
    print(f"Found {len(run_ids)} experimental runs: {run_ids}")
    print(
        f"alpha={cfg.alpha}  pers_quantile={cfg.persistence_samples}  pers_adjust={cfg.pers_adjust}",
        flush=True,
    )

    onsets: dict[int, int] = {}
    n_workers = multiprocessing.cpu_count() if args.workers == -1 else max(1, args.workers)
    tasks = [(run_id, args.exp_dir, cfg) for run_id in run_ids]

    if n_workers <= 1:
        for task_args in tasks:
            run_id, onset = _label_one_worker(task_args)
            if onset is not None:
                onsets[run_id] = onset
    else:
        print(f"Using {n_workers} worker(s) for experimental relabelling")
        done = 0
        with multiprocessing.Pool(processes=min(n_workers, len(tasks))) as pool:
            for run_id, onset in pool.imap_unordered(_label_one_worker, tasks, chunksize=1):
                done += 1
                if onset is not None:
                    onsets[run_id] = onset
                if done % 4 == 0 or done == len(tasks):
                    print(f"  completed {done}/{len(tasks)} runs")

    print(f"\n=== Results ({len(onsets)}/{len(run_ids)} labelled) ===")
    for rid in run_ids:
        if rid in onsets:
            print(f"  run {rid:2d}: onset={onsets[rid]}")
        else:
            print(f"  run {rid:2d}: FAILED")

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in sorted(onsets.items())}, fh, indent=2)
    print(f"\nSaved → {args.out_json}")


if __name__ == "__main__":
    main()
