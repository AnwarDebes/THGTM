"""Tests for the single-layer ETTA Tsetlin Machine."""

import numpy as np
import pytest

from thgtm import EttaTsetlinMachine, MultiClassEttaTsetlinMachine, TMConfig


def make_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int8)
    return X, y


def make_noisy_xor(n=2000, noise=0.10, seed=0):
    """3-feature noisy XOR: y = x0 XOR x1, x2 is irrelevant.  Label is flipped
    with probability ``noise``."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 3)).astype(np.int8)
    y_true = X[:, 0] ^ X[:, 1]
    flip = rng.random(n) < noise
    y = np.where(flip, 1 - y_true, y_true).astype(np.int8)
    return X, y


def test_lambda_zero_trains_to_high_accuracy_on_xor():
    X, y = make_xor(n=400, seed=0)
    cfg = TMConfig(n_clauses=20, n_features=2, threshold=10, s=3.0,
                   lambda_decay=0.0, trace_alpha=0.0, seed=42)
    tm = EttaTsetlinMachine(cfg)
    tm.fit(X, y, epochs=30)
    acc = tm.score(X, y)
    assert acc >= 0.95, f"Expected >=0.95 on clean XOR, got {acc}"


def test_noisy_xor_still_high():
    X, y = make_noisy_xor(n=2000, noise=0.05, seed=1)
    cfg = TMConfig(n_clauses=40, n_features=3, threshold=15, s=3.9,
                   lambda_decay=0.0, trace_alpha=0.0, seed=7)
    tm = EttaTsetlinMachine(cfg)
    tm.fit(X, y, epochs=30)
    acc = tm.score(X, y)
    # We expect to fit close to (1 - noise) since clauses can't beat label noise.
    assert acc >= 0.85, f"Expected >=0.85 on noisy XOR, got {acc}"


def test_lambda_zero_matches_seed_run():
    """Two TMs with identical seeds + lambda=0 + alpha=0 should produce
    bit-identical state after training."""
    X, y = make_xor(n=200, seed=0)
    cfg1 = TMConfig(n_clauses=10, n_features=2, threshold=10, s=3.0,
                    lambda_decay=0.0, trace_alpha=0.0, seed=11)
    cfg2 = TMConfig(n_clauses=10, n_features=2, threshold=10, s=3.0,
                    lambda_decay=0.0, trace_alpha=0.0, seed=11)
    tm1 = EttaTsetlinMachine(cfg1).fit(X, y, epochs=5)
    tm2 = EttaTsetlinMachine(cfg2).fit(X, y, epochs=5)
    assert np.array_equal(tm1.bank.state, tm2.bank.state)


def test_multiclass_three_buckets():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.integers(0, 2, size=(n, 4)).astype(np.int8)
    # Bucketise by parity of (x0+x1) and (x2+x3)
    y = ((X[:, 0] + X[:, 1]) % 2) * 1 + ((X[:, 2] + X[:, 3]) % 2) * 2  # 0..3
    # Drop class 3 to keep 3 classes
    keep = y < 3
    X, y = X[keep], y[keep]
    mc = MultiClassEttaTsetlinMachine(
        n_classes=3, n_clauses_per_class=30, n_features=4,
        threshold=15, s=3.9, lambda_decay=0.0, trace_alpha=0.0, seed=0,
    )
    mc.fit(X, y, epochs=15)
    acc = mc.score(X, y)
    assert acc >= 0.85, f"Multi-class accuracy too low: {acc}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
