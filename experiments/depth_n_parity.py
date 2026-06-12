"""Depth-N-Parity-on-Path experiment.

This is the load-bearing test for THGTM's central claim that ETTA's
trace-projected feedback enables joint L>=2 training where vanilla HGTM
(no eligibility) collapses.

Task
----
Each sample is a path of ``path_len`` nodes carrying random {0,1} bits.
We flatten the path into a length-``path_len`` Boolean feature vector;
the label is the XOR (parity) of every bit.

For long paths a single-layer TM with a fixed clause budget cannot store
all 2^(path_len-1) parity-equivalent patterns, so the L=2 stacking should
help -- IF the layers can be trained jointly.  Vanilla HGTM's
``compute_projected_feedback`` heuristic empirically fails (its own
``benchmarks.md`` documents this).  ETTA's ``trace_projected_feedback``
is the proposed fix.

For each path_len we compare:
    M1: L=1 THGTM with the same total clause budget as L=2.
    M2: L=2 THGTM with vanilla credit assignment (alpha=0, lambda=0).
    M3: L=2 THGTM with ETTA (alpha=2.0, lambda=0.5).

Reports mean +- std test accuracy across seeds and writes a JSON to
``results/depth_n_parity.json``.

This script is deliberately small enough to run on a single CPU in a few
minutes.  All numbers in the paper come from this run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from thgtm import THGTM, THGTMConfig, LayerConfig


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_parity_dataset(n_samples: int, path_len: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, path_len)).astype(np.int8)
    y = (X.sum(axis=1) % 2).astype(np.int8)
    return X, y


def split_train_test(X, y, frac_train=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_train = int(frac_train * n)
    return (X[idx[:n_train]], y[idx[:n_train]],
            X[idx[n_train:]], y[idx[n_train:]])


def run_one_config(name, *, X_tr, y_tr, X_te, y_te,
                   n_features, n_clauses_per_layer, threshold, s,
                   layers, epochs, seed):
    """``layers`` is a list of (lambda_decay, trace_alpha) tuples (one per
    layer).  Length determines L."""
    lcs = []
    for lam, alpha in layers:
        lcs.append(LayerConfig(
            n_clauses=n_clauses_per_layer,
            n_states_per_action=100,
            threshold=threshold,
            s=s,
            lambda_decay=lam,
            trace_alpha=alpha,
        ))
    cfg = THGTMConfig(n_classes=2, n_features=n_features, layers=lcs, seed=seed)
    m = THGTM(cfg)
    t0 = time.time()
    m.fit(X_tr, y_tr, epochs=epochs)
    train_time = time.time() - t0
    train_acc = m.score(X_tr, y_tr)
    test_acc = m.score(X_te, y_te)
    return {"name": name, "train_acc": train_acc, "test_acc": test_acc,
            "train_time_s": train_time}


def main(out_path: Path = RESULTS_DIR / "depth_n_parity.json",
         path_lens=(2, 3, 4, 5),
         seeds=(0, 1, 2),
         n_samples=600,
         epochs=20):
    results = []
    overall_t0 = time.time()
    for path_len in path_lens:
        for seed in seeds:
            X, y = make_parity_dataset(n_samples=n_samples,
                                       path_len=path_len, seed=seed)
            X_tr, y_tr, X_te, y_te = split_train_test(X, y, seed=seed)
            n_features = path_len
            # Match clause budgets fairly: M1 (L=1) gets 2x the per-layer
            # budget so total clauses are comparable to L=2.
            per_layer = 24
            l1 = run_one_config(
                "L1",
                X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                n_features=n_features,
                n_clauses_per_layer=per_layer * 2, threshold=12, s=3.0,
                layers=[(0.0, 0.0)],
                epochs=epochs, seed=seed,
            )
            l2_no = run_one_config(
                "L2_no_etta",
                X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                n_features=n_features,
                n_clauses_per_layer=per_layer, threshold=12, s=3.0,
                layers=[(0.0, 0.0), (0.0, 0.0)],
                epochs=epochs, seed=seed,
            )
            l2_etta = run_one_config(
                "L2_etta",
                X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                n_features=n_features,
                n_clauses_per_layer=per_layer, threshold=12, s=3.0,
                layers=[(0.5, 2.0), (0.5, 2.0)],
                epochs=epochs, seed=seed,
            )
            for row in (l1, l2_no, l2_etta):
                row.update({"path_len": path_len, "seed": seed})
                results.append(row)
                print(f"path_len={path_len} seed={seed} {row['name']}: "
                      f"train={row['train_acc']:.3f} test={row['test_acc']:.3f} "
                      f"({row['train_time_s']:.1f}s)")
    total = time.time() - overall_t0
    print(f"\nTotal: {total:.1f}s")

    # Summary
    summary = {}
    names = sorted({r["name"] for r in results})
    for pl in path_lens:
        summary[pl] = {}
        for n in names:
            rows = [r for r in results if r["path_len"] == pl and r["name"] == n]
            tas = np.array([r["test_acc"] for r in rows])
            summary[pl][n] = {"mean": float(tas.mean()),
                              "std": float(tas.std()),
                              "n_seeds": len(rows)}
    print("\n=== SUMMARY (test accuracy mean +- std) ===")
    for pl, by_name in summary.items():
        line = f"path_len={pl}:  "
        for n in names:
            v = by_name[n]
            line += f"{n}={v['mean']:.3f}+-{v['std']:.3f}  "
        print(line)

    payload = {"results": results, "summary": summary,
               "config": {"path_lens": list(path_lens),
                          "seeds": list(seeds),
                          "n_samples": n_samples,
                          "epochs": epochs}}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
