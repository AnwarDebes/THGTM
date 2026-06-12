"""Echo-Trace Tsetlin Automata (ETTA).

The core primitive of THGTM.  A standard 2-action Tsetlin Automaton with N
states per action (2N total) augmented with an exponentially-decaying
eligibility trace.

    state in {1, ..., 2N}   -> include iff state > N
    trace in [0, 1]         -> decayed memory of state transitions

When lambda_decay == 0 and trace_alpha == 0 the bank reduces to a vanilla
TA bank (Granmo 2018) bit-exactly given the same RNG stream.

The trace is the load-bearing addition that enables (a) sequence learning
and (b) multi-layer credit assignment in stacked GraphTMs (THGTM).
"""

from __future__ import annotations

import numpy as np


class ETTABank:
    """Vectorised bank of Echo-Trace Tsetlin Automata.

    One TA per (clause, literal) pair.

    Parameters
    ----------
    n_clauses : int
        Number of clauses.
    n_literals : int
        Number of literals (typically 2 * n_features for raw + negated).
    n_states_per_action : int
        Number of states per action (the 'N' parameter).  Total states 2N.
    lambda_decay : float in [0, 1)
        Trace decay factor per timestep.  0 disables the trace.
    trace_alpha : float >= 0
        Multiplier on trace contribution to feedback amplification.
        0 disables trace effect on update probabilities.
    rng : np.random.Generator | None
        RNG; if None a fresh default_rng is created.
    """

    def __init__(
        self,
        n_clauses: int,
        n_literals: int,
        n_states_per_action: int = 100,
        lambda_decay: float = 0.0,
        trace_alpha: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        if not (0.0 <= lambda_decay < 1.0):
            raise ValueError("lambda_decay must be in [0, 1)")
        if trace_alpha < 0.0:
            raise ValueError("trace_alpha must be >= 0")

        self.n_clauses = int(n_clauses)
        self.n_literals = int(n_literals)
        self.N = int(n_states_per_action)
        self.lam = float(lambda_decay)
        self.alpha = float(trace_alpha)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Start at the exclude boundary, exactly as the canonical TM does.
        self.state = np.full(
            (self.n_clauses, self.n_literals), self.N, dtype=np.int32
        )
        # Trace per TA in [0, 1].
        self.trace = np.zeros((self.n_clauses, self.n_literals), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Read API
    # ------------------------------------------------------------------ #
    def actions(self) -> np.ndarray:
        """Return (n_clauses, n_literals) bool: True iff TA includes literal."""
        return self.state > self.N

    def clause_outputs(self, literals: np.ndarray, predict: bool = False) -> np.ndarray:
        """Evaluate every clause on a single (1D) literal vector.

        Parameters
        ----------
        literals : (n_literals,) bool / 0-1 array
            Literal values for this sample (raw + negated, in the caller's order).
        predict : bool
            In predict mode an all-exclude clause outputs 0.  In train mode it
            outputs 1.  (Standard Granmo convention.)

        Returns
        -------
        (n_clauses,) int8 array of clause outputs.
        """
        literals = np.asarray(literals, dtype=bool)
        inc = self.actions()                                 # (C, L)
        # For each clause: clause = AND over included literals of literal_value.
        # Equivalently: clause = 1 iff there is no included literal that is 0.
        bad = inc & ~literals[np.newaxis, :]                 # included AND value=0
        any_bad = bad.any(axis=1)                            # (C,)
        out = (~any_bad).astype(np.int8)
        if predict:
            empty = ~inc.any(axis=1)
            out = np.where(empty, np.int8(0), out)
        return out

    # ------------------------------------------------------------------ #
    # Core feedback
    # ------------------------------------------------------------------ #
    def _apply_delta(self, delta: np.ndarray) -> None:
        """Apply integer state deltas (n_clauses, n_literals) in {-1, 0, +1}.

        Updates traces by trace = lam * trace + |delta|, clamped to 1, only
        when ``lambda_decay > 0``.  With ``lambda_decay == 0`` the trace stays
        identically zero, matching vanilla TA behaviour bit-exactly under the
        same RNG stream.
        """
        np.clip(self.state + delta, 1, 2 * self.N, out=self.state)
        if self.lam > 0.0:
            np.multiply(self.trace, self.lam, out=self.trace)
            if (delta != 0).any():
                mag = (delta != 0).astype(np.float32)
                self.trace += mag
                np.minimum(self.trace, 1.0, out=self.trace)

    def trace_step(self) -> None:
        """Apply pure trace decay (no state change).  Use between samples
        in sequence learning."""
        if self.lam > 0.0:
            np.multiply(self.trace, self.lam, out=self.trace)

    def type_I_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        s: float,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        """Apply Type I (boost-and-forget) feedback to selected clauses.

        Parameters
        ----------
        clause_outputs : (n_clauses,) array of 0/1
            Output of every clause on the current sample.
        literals : (n_literals,) array of 0/1
            Current literal vector.
        s : float > 1
            Specificity.  Larger s -> sparser, more specific clauses.
        feedback_mask : (n_clauses,) bool or None
            If given, only clauses with True receive updates; others are left
            untouched.  None means all clauses (legacy behaviour).
        """
        c = np.asarray(clause_outputs, dtype=bool)
        x = np.asarray(literals, dtype=bool)
        C, L = self.n_clauses, self.n_literals

        boost_p = (s - 1.0) / s
        forget_p = 1.0 / s
        if self.alpha > 0.0:
            # Trace-amplified probabilities: recently-active TAs see larger
            # update probabilities, capped at 1.0.  amp >= 1 always since
            # trace and alpha are non-negative.
            amp = 1.0 + self.alpha * self.trace
            boost_p_arr = np.minimum(1.0, boost_p * amp)
            forget_p_arr = np.minimum(1.0, forget_p * amp)
        else:
            boost_p_arr = np.full((C, L), boost_p, dtype=np.float32)
            forget_p_arr = np.full((C, L), forget_p, dtype=np.float32)

        boost_mask = c[:, None] & x[None, :]            # clause fires AND literal=1
        forget_mask = ~boost_mask                       # everything else

        rand = self.rng.random((C, L)).astype(np.float32)
        delta = np.zeros((C, L), dtype=np.int32)
        delta[boost_mask & (rand < boost_p_arr)] = 1
        delta[forget_mask & (rand < forget_p_arr)] = -1

        if feedback_mask is not None:
            fm = np.asarray(feedback_mask, dtype=bool)
            delta[~fm, :] = 0

        self._apply_delta(delta)

    def type_II_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        """Apply Type II (discriminative) feedback to selected clauses.

        Type II is deterministic: if clause fires AND literal=0 AND TA excludes,
        push state toward include (+1).  Nothing else changes.
        """
        c = np.asarray(clause_outputs, dtype=bool)
        x = np.asarray(literals, dtype=bool)
        inc = self.actions()                                    # (C, L)
        mask = c[:, None] & (~x[None, :]) & (~inc)
        delta = mask.astype(np.int32)
        if feedback_mask is not None:
            fm = np.asarray(feedback_mask, dtype=bool)
            delta[~fm, :] = 0
        self._apply_delta(delta)

    # ------------------------------------------------------------------ #
    # Multi-layer credit assignment via traces  (the THGTM novelty)
    # ------------------------------------------------------------------ #
    def trace_projected_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        downstream_credit: np.ndarray,
        s: float,
    ) -> None:
        """Trace-projected feedback for an intermediate layer.

        Standard HGTM uses a heuristic ``compute_projected_feedback`` kernel
        that aggregates downstream class-clause-update signs.  That kernel
        collapses at L >= 2 because at initialisation no downstream clause
        fires, so credit is zero, so layer 1 never starts.

        ETTA fixes this:  ``downstream_credit[c]`` carries the signed credit
        for clause c (positive => Type-I-like reward, negative => Type-II-like
        penalty), and the trace per TA scales how much of that credit actually
        translates into a state delta.  Even at initialisation, traces seeded
        by tiny exploratory drift create a non-zero gradient.

        Parameters
        ----------
        clause_outputs : (n_clauses,)
            Output of THIS layer's clauses on the current sample.
        literals : (n_literals,)
            THIS layer's input literals.
        downstream_credit : (n_clauses,) float in [-1, 1]
            Signed credit signal coming from the next layer or class head.
        s : float
            Specificity (same as Type I).
        """
        c = np.asarray(clause_outputs, dtype=bool)
        x = np.asarray(literals, dtype=bool)
        credit = np.asarray(downstream_credit, dtype=np.float32)
        C, L = self.n_clauses, self.n_literals

        # Eligibility = base 1.0 (so |credit| alone drives learning) plus
        # additive amplification by the trace.  This is the key design choice
        # that breaks the chicken-and-egg at HGTM init: when trace == 0 the
        # update probability is still |credit| * (s-1)/s, exactly what a
        # standard layer would compute under direct supervision.  When trace
        # grows, recently-active TAs see proportionally larger updates.
        eligibility = 1.0 + self.alpha * self.trace               # (C, L) in [1, 1+alpha]

        eff = credit[:, None] * eligibility                       # (C, L) signed

        # Positive credit -> Type-I-like behaviour scaled by |eff|.
        boost_p = np.clip(np.where(eff > 0.0, eff, 0.0) * (s - 1.0) / s, 0.0, 1.0)
        forget_p = np.clip(np.where(eff > 0.0, eff, 0.0) / s, 0.0, 1.0)
        # Negative credit -> Type-II-like: discriminate where clause fires
        # despite literal=0 and TA excludes.
        penal_p = np.clip(np.where(eff < 0.0, -eff, 0.0), 0.0, 1.0)

        boost_mask = c[:, None] & x[None, :]
        forget_mask = ~boost_mask
        inc = self.actions()
        penal_mask = c[:, None] & (~x[None, :]) & (~inc)

        rand = self.rng.random((C, L)).astype(np.float32)
        delta = np.zeros((C, L), dtype=np.int32)
        delta[boost_mask & (rand < boost_p)] = 1
        delta[forget_mask & (rand < forget_p)] = -1
        # Penalty is overridden last so it can flip a forget into a boost.
        delta[penal_mask & (rand < penal_p)] = 1

        self._apply_delta(delta)


# A tiny convenience class for single-TA experimentation.
class EchoTraceAutomaton:
    """Single Echo-Trace Tsetlin Automaton.  Useful for unit tests and diagrams."""

    def __init__(self, n_states_per_action: int = 100, lambda_decay: float = 0.0):
        self.N = int(n_states_per_action)
        self.lam = float(lambda_decay)
        self.state = self.N
        self.trace = 0.0

    @property
    def includes(self) -> bool:
        return self.state > self.N

    def update(self, delta: int) -> None:
        self.state = max(1, min(2 * self.N, self.state + delta))
        self.trace = self.lam * self.trace + (1.0 if delta != 0 else 0.0)
        self.trace = min(self.trace, 1.0)

    def step(self) -> None:
        self.trace = self.lam * self.trace
