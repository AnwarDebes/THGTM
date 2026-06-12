"""Tests for trajectory receipts."""

import numpy as np
import pytest

from thgtm import (
    ClauseReceipt, TrajectoryReceipt, LTLSkeleton,
    build_clause_receipt, verify_trajectory,
)


def test_per_step_receipt_replays():
    includes_mask = np.array([1, 0, 1, 0], dtype=bool)
    literals = np.array([1, 0, 1, 1], dtype=np.int8)
    r = build_clause_receipt(step=0, clause_id=7, layer=0,
                             includes_mask=includes_mask, literals=literals)
    assert r.output == 1                          # 1 AND 1 = 1
    ok, info = verify_trajectory(TrajectoryReceipt(receipts=[r]))
    assert ok
    assert info["per_step_ok"] == [True]


def test_per_step_failure_detected():
    includes_mask = np.array([1, 0, 1, 0], dtype=bool)
    literals = np.array([1, 0, 1, 1], dtype=np.int8)
    r = build_clause_receipt(step=0, clause_id=7, layer=0,
                             includes_mask=includes_mask, literals=literals)
    # Mutate the stored literals to attempt forgery.
    r.literals[2] = 0
    ok, info = verify_trajectory(TrajectoryReceipt(receipts=[r]))
    assert not ok                                # detection succeeds


def test_hmac_signature_round_trip():
    key = b"test-key-1234567890"
    includes_mask = np.array([1, 1], dtype=bool)
    literals = np.array([1, 1], dtype=np.int8)
    r = build_clause_receipt(step=0, clause_id=0, layer=0,
                             includes_mask=includes_mask, literals=literals,
                             key=key)
    assert r.signature is not None
    assert r.verify_signature(key)
    # Tampering invalidates the signature
    r.output = 0
    assert not r.verify_signature(key)


def test_ltl_eventually_within():
    # Fire / no-fire / fire over 3 steps; the LTL says "must fire in last 2".
    rs = []
    for step, fired in enumerate([1, 0, 1]):
        lits = np.array([1] if fired else [0], dtype=np.int8)
        inc = np.array([1], dtype=bool)
        rs.append(build_clause_receipt(step, 0, 0, inc, lits))
    ltl = LTLSkeleton(constraints=[("eventually_within", 0, 2)])
    ok, info = verify_trajectory(TrajectoryReceipt(receipts=rs, ltl=ltl))
    assert ok


def test_ltl_count_le_violation():
    rs = []
    for step in range(5):
        lits = np.array([1], dtype=np.int8)
        inc = np.array([1], dtype=bool)
        rs.append(build_clause_receipt(step, 0, 0, inc, lits))
    # All five fire; cap at 2 -> violation.
    ltl = LTLSkeleton(constraints=[("count_le", 0, 2, 5)])
    ok, info = verify_trajectory(TrajectoryReceipt(receipts=rs, ltl=ltl))
    assert not ok
    assert any("count_le" in v for v in info["ltl_violations"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
