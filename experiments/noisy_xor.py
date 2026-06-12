"""Noisy-XOR sanity experiment.

Verifies that with ``lambda_decay = 0`` and ``trace_alpha = 0`` the
ETTA-augmented Tsetlin Machine is empirically indistinguishable from a
vanilla TM, recovering canonical Noisy-XOR accuracy (~98-99% under low
label noise).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from thgtm.tm import EttaTsetlinMachine, TMConfig


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_noisy_xor(n=2000, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 3)).astype(np.int8)
    y_true = X[:, 0] ^ X[:, 1]
    flip = rng.random(n) < noise
    y = np.where(flip, 1 - y_true, y_true).astype(np.int8)
    return X, y, y_true


def main(out_path: Path = RESULTS_DIR / "noisy_xor.json",
         seeds=(0, 1, 2, 3, 4), noise=0.05, epochs=30):
    results = []
    for seed in seeds:
        Xtr, ytr, ytr_true = make_noisy_xor(n=2000, noise=noise, seed=seed)
        Xte, yte, yte_true = make_noisy_xor(n=500, noise=0.0, seed=seed + 1000)
        cfg = TMConfig(n_clauses=40, n_features=3, threshold=15, s=3.9,
                       lambda_decay=0.0, trace_alpha=0.0, seed=seed)
        tm = EttaTsetlinMachine(cfg)
        t0 = time.time()
        tm.fit(Xtr, ytr, epochs=epochs)
        elapsed = time.time() - t0
        clean_acc = float((tm.predict(Xte) == yte_true).mean())
        results.append({"seed": seed, "test_acc_clean": clean_acc,
                        "train_time_s": elapsed})
        print(f"seed={seed} clean_test_acc={clean_acc:.3f} ({elapsed:.1f}s)")

    accs = np.array([r["test_acc_clean"] for r in results])
    print(f"\nMean clean test acc: {accs.mean():.3f} +- {accs.std():.3f}")
    out_path.write_text(json.dumps({"results": results,
                                    "summary": {"mean": float(accs.mean()),
                                                "std": float(accs.std())}},
                                   indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
