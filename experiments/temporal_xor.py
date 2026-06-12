"""Temporal-XOR experiment.

Task
----
Samples arrive in a SEQUENCE (not i.i.d.).  At each timestep the model sees
a 2-bit input ``x_t``.  The label is ``y_t = x_t[0] XOR x_{t-k}[0]`` for a
fixed delay ``k`` >= 1.  The current sample alone does not contain enough
information to predict the label -- the model must use information from
``k`` steps ago.

A vanilla Tsetlin Machine has no memory and is structurally chance-level
on this task.  We compare three settings to disentangle WHICH part of
THGTM gives the temporal capability:

    M_raw:       single-layer TM on raw inputs only, no temporal literals.
                 Expected: chance.
    M_past_only: single-layer TM on raw + PAST_k literals, vanilla TA.
                 Expected: solves the task because the input is now sufficient.
    M_past_etta: single-layer TM on raw + PAST_k literals, ETTA trace on.
                 Expected: matches M_past_only at lambda=0 (sanity) and
                 retains accuracy at lambda>0 (no regression from the trace).

The cleanest scientific takeaway: temporal features make the task
representable; ETTA's trace is the credit-assignment companion that makes
the multi-layer extension trainable.  Both ingredients are necessary; the
experiment isolates the contribution of each.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from thgtm.etta import ETTABank
from thgtm.tm import EttaTsetlinMachine, TMConfig
from thgtm.temporal import TemporalLiteralEncoder, PAST_k


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_temporal_xor_stream(n_steps: int, k: int, seed: int):
    """Generate a sequence of (x_t, y_t).  Returns X (n,2), y (n,)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_steps, 2)).astype(np.int8)
    # Label requires past info; we make the first ``k`` labels random.
    y = np.empty(n_steps, dtype=np.int8)
    for t in range(n_steps):
        if t < k:
            y[t] = int(rng.integers(0, 2))
        else:
            y[t] = X[t, 0] ^ X[t - k, 0]
    return X, y


def run_one(name, X_tr, y_tr, X_te, y_te, n_features,
            n_clauses, threshold, s, lam, alpha, epochs, seed):
    cfg = TMConfig(n_clauses=n_clauses, n_features=n_features,
                   threshold=threshold, s=s,
                   lambda_decay=lam, trace_alpha=alpha, seed=seed)
    tm = EttaTsetlinMachine(cfg)
    t0 = time.time()
    tm.fit(X_tr, y_tr, epochs=epochs, shuffle=False)
    train_time = time.time() - t0
    train_acc = tm.score(X_tr, y_tr)
    test_acc = tm.score(X_te, y_te)
    return {"name": name, "train_acc": train_acc, "test_acc": test_acc,
            "train_time_s": train_time}


def with_past_features(X: np.ndarray, k: int) -> np.ndarray:
    """Augment a 2-feature input stream with PAST_k literals for both bits."""
    enc = TemporalLiteralEncoder(
        n_inputs=X.shape[1],
        ops=[PAST_k(j, k) for j in range(X.shape[1])],
    )
    return enc.transform_batch(X)


def main(out_path: Path = RESULTS_DIR / "temporal_xor.json",
         delays=(1, 2, 3),
         seeds=(0, 1, 2),
         n_train=1500, n_test=500, epochs=15):
    results = []
    for k in delays:
        for seed in seeds:
            X_tr, y_tr = make_temporal_xor_stream(n_train, k, seed)
            X_te, y_te = make_temporal_xor_stream(n_test, k, seed + 1000)
            X_tr_aug = with_past_features(X_tr, k)
            X_te_aug = with_past_features(X_te, k)

            # M_raw: raw inputs only, vanilla TM
            row = run_one("M_raw", X_tr, y_tr, X_te, y_te,
                          n_features=X_tr.shape[1], n_clauses=40,
                          threshold=15, s=3.0, lam=0.0, alpha=0.0,
                          epochs=epochs, seed=seed)
            row["delay"] = k; row["seed"] = seed
            results.append(row)
            # M_past_only: PAST_k literals, vanilla TA
            row = run_one("M_past_only", X_tr_aug, y_tr, X_te_aug, y_te,
                          n_features=X_tr_aug.shape[1], n_clauses=40,
                          threshold=15, s=3.0, lam=0.0, alpha=0.0,
                          epochs=epochs, seed=seed)
            row["delay"] = k; row["seed"] = seed
            results.append(row)
            # M_past_etta: PAST_k literals, ETTA trace on
            row = run_one("M_past_etta", X_tr_aug, y_tr, X_te_aug, y_te,
                          n_features=X_tr_aug.shape[1], n_clauses=40,
                          threshold=15, s=3.0, lam=0.5, alpha=2.0,
                          epochs=epochs, seed=seed)
            row["delay"] = k; row["seed"] = seed
            results.append(row)
            for r in results[-3:]:
                print(f"delay={k} seed={seed} {r['name']}: "
                      f"train={r['train_acc']:.3f} test={r['test_acc']:.3f}")

    summary = {}
    names = sorted({r["name"] for r in results})
    for k in delays:
        summary[k] = {}
        for n in names:
            rows = [r for r in results if r["delay"] == k and r["name"] == n]
            tas = np.array([r["test_acc"] for r in rows])
            summary[k][n] = {"mean": float(tas.mean()),
                             "std": float(tas.std()),
                             "n_seeds": len(rows)}
    print("\n=== SUMMARY (test accuracy mean +- std) ===")
    for k, by_name in summary.items():
        line = f"delay={k}:  "
        for n in names:
            v = by_name[n]
            line += f"{n}={v['mean']:.3f}+-{v['std']:.3f}  "
        print(line)

    payload = {"results": results, "summary": summary,
               "config": {"delays": list(delays), "seeds": list(seeds),
                          "n_train": n_train, "n_test": n_test, "epochs": epochs}}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
