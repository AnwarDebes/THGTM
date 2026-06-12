"""Tests for the bounded-LTL temporal literal encoder."""

import numpy as np
import pytest

from thgtm.temporal import (
    TemporalLiteralEncoder, PAST_k, SINCE, ALWAYS_in_window,
)


def test_past_k_returns_zero_before_history():
    enc = TemporalLiteralEncoder(n_inputs=1, ops=[PAST_k(0, 2)])
    out = enc.transform(np.array([1], dtype=np.int8))
    # Only one sample in history; PAST_2 should be zero.
    assert out[1] == 0


def test_past_k_returns_correct_value_after_history():
    enc = TemporalLiteralEncoder(n_inputs=1, ops=[PAST_k(0, 2)])
    enc.transform(np.array([1], dtype=np.int8))
    enc.transform(np.array([0], dtype=np.int8))
    out = enc.transform(np.array([0], dtype=np.int8))
    # PAST_2 from now = the first sample = 1
    assert out[1] == 1


def test_always_in_window_basic():
    enc = TemporalLiteralEncoder(n_inputs=1, ops=[ALWAYS_in_window(0, 3)])
    enc.transform(np.array([1], dtype=np.int8))
    enc.transform(np.array([1], dtype=np.int8))
    out = enc.transform(np.array([1], dtype=np.int8))
    assert out[1] == 1
    out = enc.transform(np.array([0], dtype=np.int8))
    # window [t-2, t-1, t] = [1, 1, 0] -> not always
    assert out[1] == 0


def test_since_basic():
    enc = TemporalLiteralEncoder(n_inputs=2, ops=[SINCE(0, 1, 5)])
    # Sequence: a=1 at t=0, then b=1 holds, asks SINCE
    enc.transform(np.array([1, 1], dtype=np.int8))      # t=0: a=1 -> trivially since
    enc.transform(np.array([0, 1], dtype=np.int8))      # t=1: b held since t=0
    out = enc.transform(np.array([0, 1], dtype=np.int8))  # t=2: b held since t=0
    assert out[2] == 1
    out = enc.transform(np.array([0, 0], dtype=np.int8))  # t=3: b broke
    assert out[2] == 0


def test_transform_batch_matches_sequential():
    enc1 = TemporalLiteralEncoder(n_inputs=2, ops=[PAST_k(1, 1)])
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(20, 2)).astype(np.int8)
    seq = np.array([enc1.transform(X[i]) for i in range(20)])

    enc2 = TemporalLiteralEncoder(n_inputs=2, ops=[PAST_k(1, 1)])
    bat = enc2.transform_batch(X)
    assert np.array_equal(seq, bat)


def test_reset():
    enc = TemporalLiteralEncoder(n_inputs=1, ops=[PAST_k(0, 1)])
    enc.transform(np.array([1], dtype=np.int8))
    enc.reset()
    out = enc.transform(np.array([0], dtype=np.int8))
    assert out[1] == 0  # PAST_1 had no history again


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
