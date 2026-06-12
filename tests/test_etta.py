"""Unit tests for the ETTA primitive."""

import numpy as np
import pytest

from thgtm.etta import ETTABank, EchoTraceAutomaton


def test_initial_actions_all_exclude():
    bank = ETTABank(n_clauses=4, n_literals=10, n_states_per_action=100)
    # State starts at N which is the exclude boundary (need state > N to include).
    assert not bank.actions().any()


def test_lambda_zero_keeps_trace_zero():
    rng = np.random.default_rng(0)
    bank = ETTABank(4, 10, n_states_per_action=20, lambda_decay=0.0, rng=rng)
    literals = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    co = bank.clause_outputs(literals)
    bank.type_I_feedback(co, literals, s=3.9)
    # With lam=0 the trace is forced back to zero each step, regardless of
    # whether deltas were non-zero.
    assert bank.trace.sum() == 0.0


def test_lambda_positive_grows_then_decays():
    rng = np.random.default_rng(0)
    bank = ETTABank(4, 10, n_states_per_action=20, lambda_decay=0.5, rng=rng)
    literals = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    for _ in range(5):
        co = bank.clause_outputs(literals)
        bank.type_I_feedback(co, literals, s=3.9)
    nonzero_after_train = float(bank.trace.sum())
    assert nonzero_after_train > 0.0
    for _ in range(20):
        bank.trace_step()
    assert float(bank.trace.sum()) < nonzero_after_train * 1e-3


def test_clause_outputs_basic():
    bank = ETTABank(2, 4, n_states_per_action=10)
    # Manually flip a TA to include
    bank.state[0, 0] = bank.N + 1
    literals_yes = np.array([1, 0, 0, 0], dtype=np.int8)
    literals_no = np.array([0, 0, 0, 0], dtype=np.int8)
    out_yes = bank.clause_outputs(literals_yes, predict=False)
    out_no = bank.clause_outputs(literals_no, predict=False)
    assert out_yes[0] == 1
    assert out_no[0] == 0
    # In predict mode, clause 1 has no inclusions -> output 0
    out_predict = bank.clause_outputs(literals_yes, predict=True)
    assert out_predict[1] == 0


def test_type_I_feedback_increments_when_clause_and_literal_fire():
    rng = np.random.default_rng(0)
    bank = ETTABank(1, 1, n_states_per_action=100, lambda_decay=0.0, rng=rng)
    # Force the TA to include so clause output is 1
    bank.state[0, 0] = bank.N + 1
    literals = np.array([1], dtype=np.int8)
    co = bank.clause_outputs(literals)
    assert co[0] == 1
    start = bank.state[0, 0]
    # Run several updates; expectation is that state moves toward 2N (boost).
    for _ in range(200):
        bank.type_I_feedback(co, literals, s=3.9)
    assert bank.state[0, 0] > start


def test_type_II_feedback_includes_when_clause_fires_with_literal_zero():
    """Type II should drive an excluded TA toward include when clause=1, lit=0."""
    rng = np.random.default_rng(0)
    bank = ETTABank(1, 2, n_states_per_action=100, lambda_decay=0.0, rng=rng)
    # TA[0,0] includes literal 0; TA[0,1] excludes literal 1.
    bank.state[0, 0] = bank.N + 1
    bank.state[0, 1] = bank.N - 5
    literals = np.array([1, 0], dtype=np.int8)
    co = bank.clause_outputs(literals)
    assert co[0] == 1                       # clause fires
    s_before = bank.state[0, 1]
    bank.type_II_feedback(co, literals)
    assert bank.state[0, 1] == s_before + 1


def test_trace_projected_feedback_does_something_at_init():
    """Without ETTA's trace floor, projected feedback at init is exactly zero
    (the chicken-and-egg failure mode in HGTM).  Verify that ETTA gives a
    non-zero update at the very first step."""
    rng = np.random.default_rng(0)
    bank = ETTABank(3, 4, n_states_per_action=50,
                    lambda_decay=0.5, trace_alpha=1.0, rng=rng)
    literals = np.array([1, 0, 1, 0], dtype=np.int8)
    co = bank.clause_outputs(literals)
    state_before = bank.state.copy()
    credit = np.array([0.5, -0.3, 0.0], dtype=np.float32)
    bank.trace_projected_feedback(co, literals, credit, s=3.9)
    assert not np.array_equal(bank.state, state_before), (
        "ETTA must produce a non-zero update at init thanks to the trace "
        "floor 1/N; otherwise the L>=2 chicken-and-egg bug persists."
    )


def test_echo_trace_automaton_solo():
    a = EchoTraceAutomaton(n_states_per_action=5, lambda_decay=0.5)
    assert not a.includes
    for _ in range(6):
        a.update(+1)
    assert a.includes
    # Decay
    for _ in range(50):
        a.step()
    assert a.trace < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
