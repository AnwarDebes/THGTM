"""Tests for the THGTM multi-layer architecture."""

import numpy as np
import pytest

from thgtm import THGTM, THGTMConfig, LayerConfig


def make_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int8)
    return X, y


def test_thgtm_l1_matches_xor():
    """L=1 THGTM should still learn XOR (it's effectively a multi-class TM)."""
    X, y = make_xor(n=400, seed=0)
    cfg = THGTMConfig(
        n_classes=2,
        n_features=2,
        layers=[LayerConfig(n_clauses=20, threshold=10, s=3.0,
                            lambda_decay=0.0, trace_alpha=0.0)],
        seed=42,
    )
    m = THGTM(cfg)
    m.fit(X, y, epochs=20)
    acc = m.score(X, y)
    assert acc >= 0.90, f"L=1 THGTM should solve XOR; got {acc}"


def test_thgtm_l2_runs_and_does_not_crash_on_init():
    """L=2 with eligibility traces should make at least SOME progress
    (i.e., not collapse to chance forever).  This is the load-bearing
    sanity check vs. the canonical-HGTM L>=2 failure."""
    X, y = make_xor(n=400, seed=0)
    cfg = THGTMConfig(
        n_classes=2,
        n_features=2,
        layers=[
            LayerConfig(n_clauses=20, threshold=10, s=3.0,
                        lambda_decay=0.5, trace_alpha=2.0),
            LayerConfig(n_clauses=20, threshold=10, s=3.0,
                        lambda_decay=0.5, trace_alpha=2.0),
        ],
        seed=42,
    )
    m = THGTM(cfg)
    m.fit(X, y, epochs=20)
    acc = m.score(X, y)
    # Modest bar -- we just want to confirm L=2 is not stuck at chance.
    assert acc > 0.55, f"L=2 THGTM collapsed to chance ({acc})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
