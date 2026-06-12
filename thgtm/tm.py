"""Single-layer Tsetlin Machine with ETTA.

A clean reference implementation of the binary and multi-class Tsetlin
Machine (Granmo, JMLR 2018) built on the ETTABank primitive.  When the
``lambda_decay`` and ``trace_alpha`` parameters are both zero the trainer
matches vanilla TM behaviour given the same RNG stream.

The module exposes:

* ``EttaTsetlinMachine`` -- binary classifier with ``n_clauses`` clauses.
* ``MultiClassEttaTsetlinMachine`` -- one-vs-rest wrapper over the binary
  trainer, used as the THGTM single-layer building block.

We deliberately keep everything in pure NumPy.  The whole point of the
reference implementation is reviewer-readability and reproducibility on a
laptop, not raw speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .etta import ETTABank


def to_literals(X: np.ndarray) -> np.ndarray:
    """Convert a (n_samples, n_features) 0/1 array into a (n_samples, 2*n_features)
    array of literals (positive then negated)."""
    X = np.asarray(X, dtype=np.int8)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be binary 0/1")
    return np.concatenate([X, 1 - X], axis=1)


@dataclass
class TMConfig:
    n_clauses: int
    n_features: int
    n_states_per_action: int = 100
    threshold: int = 15
    s: float = 3.9
    lambda_decay: float = 0.0
    trace_alpha: float = 0.0
    seed: int | None = None


class EttaTsetlinMachine:
    """Binary Tsetlin Machine with Echo-Trace automata.

    The ``n_clauses`` is the TOTAL number of clauses; half are positive
    polarity (vote +1 for class 1) and half are negative polarity (vote
    +1 for class 0).  ``n_clauses`` must be even.
    """

    def __init__(self, cfg: TMConfig):
        if cfg.n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even (half +polarity, half -polarity).")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.bank = ETTABank(
            n_clauses=cfg.n_clauses,
            n_literals=2 * cfg.n_features,
            n_states_per_action=cfg.n_states_per_action,
            lambda_decay=cfg.lambda_decay,
            trace_alpha=cfg.trace_alpha,
            rng=self.rng,
        )
        self.polarity = np.empty(cfg.n_clauses, dtype=np.int8)
        self.polarity[: cfg.n_clauses // 2] = 1
        self.polarity[cfg.n_clauses // 2:] = -1

    # ---------- inference ----------
    def vote_sum(self, literals_row: np.ndarray, predict: bool) -> tuple[int, np.ndarray]:
        co = self.bank.clause_outputs(literals_row, predict=predict)
        v = int((self.polarity.astype(np.int32) * co.astype(np.int32)).sum())
        return v, co

    def predict_one(self, literals_row: np.ndarray) -> int:
        v, _ = self.vote_sum(literals_row, predict=True)
        return 1 if v > 0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        lits = to_literals(X)
        return np.array([self.predict_one(lits[i]) for i in range(lits.shape[0])],
                        dtype=np.int8)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ---------- training ----------
    def _train_step(self, literals_row: np.ndarray, y: int) -> None:
        v, co = self.vote_sum(literals_row, predict=False)
        T = self.cfg.threshold
        v_clipped = max(-T, min(T, v))

        if y == 1:
            # Positive class: positive-polarity clauses are "rewarded" (Type I).
            #                negative-polarity clauses are "penalised"  (Type II).
            p_feedback_pos = (T - v_clipped) / (2.0 * T)
            p_feedback_neg = (T + v_clipped) / (2.0 * T)
            self._feedback_subset(co, literals_row, polarity=+1,
                                  feedback_type="I", p=p_feedback_pos)
            self._feedback_subset(co, literals_row, polarity=-1,
                                  feedback_type="II", p=p_feedback_neg)
        else:  # y == 0
            p_feedback_pos = (T + v_clipped) / (2.0 * T)
            p_feedback_neg = (T - v_clipped) / (2.0 * T)
            self._feedback_subset(co, literals_row, polarity=+1,
                                  feedback_type="II", p=p_feedback_pos)
            self._feedback_subset(co, literals_row, polarity=-1,
                                  feedback_type="I", p=p_feedback_neg)

    def _feedback_subset(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        polarity: int,
        feedback_type: str,
        p: float,
    ) -> None:
        """Stochastically mask which clauses (of the given polarity) get feedback,
        then apply Type I or Type II to that subset only.  Unselected clauses
        are left completely untouched (no spurious forget pulses)."""
        if p <= 0.0:
            return
        feedback_mask = np.zeros(self.cfg.n_clauses, dtype=bool)
        idx = np.where(self.polarity == polarity)[0]
        if idx.size == 0:
            return
        chosen = self.rng.random(idx.size) < p
        feedback_mask[idx[chosen]] = True
        if not feedback_mask.any():
            return
        if feedback_type == "I":
            self.bank.type_I_feedback(clause_outputs, literals, s=self.cfg.s,
                                      feedback_mask=feedback_mask)
        elif feedback_type == "II":
            self.bank.type_II_feedback(clause_outputs, literals,
                                       feedback_mask=feedback_mask)
        else:
            raise ValueError("feedback_type must be 'I' or 'II'")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> "EttaTsetlinMachine":
        lits = to_literals(X)
        y = np.asarray(y, dtype=np.int8)
        for epoch in range(epochs):
            order = np.arange(lits.shape[0])
            if shuffle:
                self.rng.shuffle(order)
            for i in order:
                self._train_step(lits[i], int(y[i]))
            if verbose:
                acc = self.score(X, y)
                print(f"[EttaTM] epoch {epoch+1}/{epochs} train_acc={acc:.4f}")
        return self


class MultiClassEttaTsetlinMachine:
    """One-vs-rest wrapper around the binary trainer."""

    def __init__(
        self,
        n_classes: int,
        n_clauses_per_class: int,
        n_features: int,
        n_states_per_action: int = 100,
        threshold: int = 15,
        s: float = 3.9,
        lambda_decay: float = 0.0,
        trace_alpha: float = 0.0,
        seed: int | None = None,
    ):
        self.n_classes = int(n_classes)
        rng_seed_base = 0 if seed is None else int(seed)
        self.tms = [
            EttaTsetlinMachine(
                TMConfig(
                    n_clauses=n_clauses_per_class,
                    n_features=n_features,
                    n_states_per_action=n_states_per_action,
                    threshold=threshold,
                    s=s,
                    lambda_decay=lambda_decay,
                    trace_alpha=trace_alpha,
                    seed=rng_seed_base + c if seed is not None else None,
                )
            )
            for c in range(n_classes)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, verbose: bool = False):
        y = np.asarray(y, dtype=np.int64)
        for c in range(self.n_classes):
            yc = (y == c).astype(np.int8)
            self.tms[c].fit(X, yc, epochs=epochs, verbose=verbose)
        return self

    def vote_sums(self, X: np.ndarray) -> np.ndarray:
        lits = to_literals(X)
        out = np.zeros((lits.shape[0], self.n_classes), dtype=np.int32)
        for c in range(self.n_classes):
            for i in range(lits.shape[0]):
                v, _ = self.tms[c].vote_sum(lits[i], predict=True)
                out[i, c] = v
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.vote_sums(X), axis=1).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())
