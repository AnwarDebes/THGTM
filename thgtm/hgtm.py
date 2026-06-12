"""THGTM: Temporal Hierarchical Graph Tsetlin Machine.

A reference implementation of a stacked Tsetlin-Machine architecture in
which:

* every layer is a multi-class TM bank built on ETTA;
* the per-layer clause activations of layer ell become the literal vector
  of layer ell + 1 (the HGTM ``encode_layer_output_as_literals`` recipe,
  a reversible bit-mapping of clause output to literal vector ---
  clause i AND its negation become the next layer's literals i and K_ell + i);
* layer ell receives feedback from layer ell + 1 (or the final class head)
  via ``ETTABank.trace_projected_feedback``, which is the load-bearing
  difference vs. canonical HGTM.

For the static (non-temporal) graph case in this paper we keep the graph
representation deliberately simple: a sample is a flat literal vector
(it can come from any node-feature encoder, including ``subword-dep-graphtm``
or BERT-attention-distilled topologies).  The architecture supports that
without modification because every layer reads literals and writes clause
activations.

Temporal handling is layered on through ``TemporalLiteralEncoder``
(see ``thgtm/temporal.py``) which prepends ``PAST_k`` / ``SINCE`` /
``ALWAYS_in_window`` literals to the layer-0 literal vector.  The ETTA
trace at every layer carries the bounded history between samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from .etta import ETTABank


@dataclass
class LayerConfig:
    n_clauses: int
    n_states_per_action: int = 100
    threshold: int = 15
    s: float = 3.9
    lambda_decay: float = 0.0
    trace_alpha: float = 0.0


@dataclass
class THGTMConfig:
    n_classes: int
    n_features: int                       # at layer 0 ('positive' literals only)
    layers: List[LayerConfig] = field(default_factory=list)
    seed: int | None = None


def _encode_layer_output_as_literals(clause_outputs: np.ndarray) -> np.ndarray:
    """The bit-exact reversible mapping that turns layer ell's clause outputs
    into layer ell+1's literal vector (raw + negated, length 2*K_ell)."""
    co = np.asarray(clause_outputs, dtype=np.int8)
    return np.concatenate([co, 1 - co])


class THGTM:
    """Temporal Hierarchical Graph Tsetlin Machine.

    Stack of ETTA-backed clause layers.  The last layer's clauses are
    polarity-weighted into a per-class vote sum; intermediate layers train
    from trace-projected feedback.
    """

    def __init__(self, cfg: THGTMConfig):
        if cfg.n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not cfg.layers:
            raise ValueError("Need at least one layer")
        if any(lc.n_clauses % cfg.n_classes != 0 for lc in cfg.layers):
            # Top layer is what matters; intermediate layers can be any size.
            pass
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Build per-layer ETTABanks.  Each layer holds ONE bank with
        # n_clauses total clauses; for the top layer we split into a per-class
        # bank of n_clauses_per_class polarity-paired clauses.
        self.layer_banks: List[ETTABank] = []
        n_input_literals = 2 * cfg.n_features
        for ell, lc in enumerate(cfg.layers[:-1]):
            bank = ETTABank(
                n_clauses=lc.n_clauses,
                n_literals=n_input_literals,
                n_states_per_action=lc.n_states_per_action,
                lambda_decay=lc.lambda_decay,
                trace_alpha=lc.trace_alpha,
                rng=self.rng,
            )
            self.layer_banks.append(bank)
            n_input_literals = 2 * lc.n_clauses   # next layer's literal count

        # Final layer: one ETTABank per class (polarity-paired internally).
        top_lc = cfg.layers[-1]
        if top_lc.n_clauses % 2 != 0:
            raise ValueError("Top layer's n_clauses must be even.")
        self.top_banks: List[ETTABank] = []
        self.top_polarity = np.empty(top_lc.n_clauses, dtype=np.int8)
        self.top_polarity[: top_lc.n_clauses // 2] = 1
        self.top_polarity[top_lc.n_clauses // 2:] = -1
        for _ in range(cfg.n_classes):
            bank = ETTABank(
                n_clauses=top_lc.n_clauses,
                n_literals=n_input_literals,
                n_states_per_action=top_lc.n_states_per_action,
                lambda_decay=top_lc.lambda_decay,
                trace_alpha=top_lc.trace_alpha,
                rng=self.rng,
            )
            self.top_banks.append(bank)

        self.threshold = top_lc.threshold
        self.s_top = top_lc.s

    # --------- forward ---------
    def forward(self, X_row: np.ndarray, predict: bool) -> dict:
        """Forward pass on a single (1D) feature vector.

        Returns a dict with keys 'literals' (list per layer), 'clause_outputs'
        (list per layer, including the top per-class bank), and 'vote_sums'.
        """
        lits = np.concatenate([X_row.astype(np.int8), 1 - X_row.astype(np.int8)])
        all_lits = [lits]
        layer_outs: List[np.ndarray] = []
        for bank in self.layer_banks:
            co = bank.clause_outputs(lits, predict=predict)
            layer_outs.append(co)
            lits = _encode_layer_output_as_literals(co)
            all_lits.append(lits)

        top_outs = []
        vote_sums = np.zeros(self.cfg.n_classes, dtype=np.int32)
        for c, bank in enumerate(self.top_banks):
            co_top = bank.clause_outputs(lits, predict=predict)
            top_outs.append(co_top)
            v = int((self.top_polarity.astype(np.int32) * co_top.astype(np.int32)).sum())
            vote_sums[c] = v

        return {
            "literals": all_lits,           # length L+1 (input literals at start)
            "intermediate_outputs": layer_outs,
            "top_outputs": top_outs,
            "vote_sums": vote_sums,
        }

    def predict_one(self, X_row: np.ndarray) -> int:
        out = self.forward(X_row, predict=True)
        return int(np.argmax(out["vote_sums"]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.int8)
        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])],
                        dtype=np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # --------- training ---------
    def _train_step(self, X_row: np.ndarray, y: int) -> None:
        out = self.forward(X_row, predict=False)
        T = self.threshold

        # ---- (1) Train top-layer banks via standard Type I / Type II ----
        # Per-class downstream credit: positive for the true class, negative for
        # all others, magnitudes derived from the clipped vote sum (same shape
        # as a vanilla TM head).
        downstream_credit_top = np.zeros(self.cfg.n_classes, dtype=np.float32)
        for c in range(self.cfg.n_classes):
            v_clip = max(-T, min(T, int(out["vote_sums"][c])))
            target = +1 if c == y else -1
            # Magnitude proportional to "distance from satisfied":
            #   true class wants v >= T   -> error = (T - v)/(2T)
            #   wrong class wants v <= -T -> error = (T + v)/(2T)
            if target == +1:
                err = (T - v_clip) / (2.0 * T)
            else:
                err = (T + v_clip) / (2.0 * T)
            downstream_credit_top[c] = float(target) * float(err)

            # Train top bank for class c
            self._train_top_bank(
                bank=self.top_banks[c],
                clause_outputs=out["top_outputs"][c],
                literals=out["literals"][-1],
                y_is_this_class=(c == y),
                v_clipped=v_clip,
            )

        # ---- (2) Train intermediate layers via ETTA's trace_projected_feedback ----
        # Project per-class credit onto per-clause credit at the top layer's INPUT
        # (i.e. last intermediate layer's output).  The projection is: a clause c
        # in the last intermediate layer receives credit equal to the mean signed
        # response of all top-bank clauses that include literal "this clause" or
        # its negation.
        if self.layer_banks:
            for ell in reversed(range(len(self.layer_banks))):
                inter_bank = self.layer_banks[ell]
                inter_co = out["intermediate_outputs"][ell]
                inter_lits = out["literals"][ell]

                # Build credit signal for this intermediate layer.
                if ell == len(self.layer_banks) - 1:
                    # Top layer's input is THIS layer's output -> aggregate over
                    # every top bank.
                    credit = self._project_credit_from_top(
                        downstream_credit_top, inter_bank.n_clauses
                    )
                else:
                    # Project from the NEXT intermediate layer (already updated
                    # below; but we use the trace state PRE-update, which is fine
                    # because banks below ell haven't been touched in this step
                    # yet).
                    next_bank = self.layer_banks[ell + 1]
                    credit = self._project_credit_from_intermediate(
                        next_bank, downstream_credit_top
                    )
                # Apply trace-projected feedback.
                inter_bank.trace_projected_feedback(
                    clause_outputs=inter_co,
                    literals=inter_lits,
                    downstream_credit=credit,
                    s=self.cfg.layers[ell].s,
                )

        # ---- (3) Trace decay between samples ----
        # (No-op for layers with lambda=0.)
        # Apply between samples in the calling loop, not here, to keep this
        # function pure (forward+credit).

    def _train_top_bank(
        self,
        bank: ETTABank,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        y_is_this_class: bool,
        v_clipped: int,
    ) -> None:
        T = self.threshold
        s = self.s_top
        # Feedback subset selection mirrors single-class TM (positive-class
        # version) -- this bank's "class" is just "is the true class".
        if y_is_this_class:
            p_pos = (T - v_clipped) / (2.0 * T)
            p_neg = (T + v_clipped) / (2.0 * T)
            self._top_bank_feedback(bank, clause_outputs, literals,
                                    polarity=+1, ftype="I", p=p_pos)
            self._top_bank_feedback(bank, clause_outputs, literals,
                                    polarity=-1, ftype="II", p=p_neg)
        else:
            p_pos = (T + v_clipped) / (2.0 * T)
            p_neg = (T - v_clipped) / (2.0 * T)
            self._top_bank_feedback(bank, clause_outputs, literals,
                                    polarity=+1, ftype="II", p=p_pos)
            self._top_bank_feedback(bank, clause_outputs, literals,
                                    polarity=-1, ftype="I", p=p_neg)

    def _top_bank_feedback(self, bank: ETTABank, clause_outputs, literals,
                           polarity: int, ftype: str, p: float):
        if p <= 0.0:
            return
        idx = np.where(self.top_polarity == polarity)[0]
        if idx.size == 0:
            return
        chosen = self.rng.random(idx.size) < p
        mask = np.zeros(bank.n_clauses, dtype=bool)
        mask[idx[chosen]] = True
        if not mask.any():
            return
        if ftype == "I":
            bank.type_I_feedback(clause_outputs, literals, s=self.s_top,
                                 feedback_mask=mask)
        else:
            bank.type_II_feedback(clause_outputs, literals, feedback_mask=mask)

    def _project_credit_from_top(
        self,
        per_class_credit: np.ndarray,
        n_inter_clauses: int,
    ) -> np.ndarray:
        """Compute per-intermediate-clause credit by aggregating signed
        contributions across top banks.

        Each top bank has ``2 * n_inter_clauses`` literal-positions; positions
        0..K-1 reference the intermediate clause output, positions K..2K-1 its
        negation.  A top clause that INCLUDES literal i AND has positive
        polarity 'wants' intermediate clause i to fire; if the top bank's class
        credit is positive (true class) we reward intermediate clauses it
        supports; if negative we penalise them.
        """
        K = n_inter_clauses
        credit = np.zeros(K, dtype=np.float32)
        for c, bank in enumerate(self.top_banks):
            inc = bank.actions()                       # (top_clauses, 2K)
            pol = self.top_polarity.astype(np.float32) # (top_clauses,)
            # Influence on raw-literal positions and negated positions.
            raw = inc[:, :K].astype(np.float32) * pol[:, None]      # +-1
            neg = -inc[:, K:].astype(np.float32) * pol[:, None]     # negation flips sign
            per_clause_inf = (raw + neg).mean(axis=0)               # (K,)
            credit += per_class_credit[c] * per_clause_inf
        # Normalise to [-1, 1]
        max_abs = float(np.max(np.abs(credit))) + 1e-9
        return credit / max_abs

    def _project_credit_from_intermediate(
        self,
        next_bank: ETTABank,
        per_class_credit: np.ndarray,
    ) -> np.ndarray:
        """Approximate projection from a non-top intermediate layer.

        For chained intermediate layers we use the same trick: a clause's
        importance to the layer above is its mean signed inclusion across the
        layer above's clauses.  The next layer's bank stores the inclusion
        state we need; per-class credit is the *current* signal we're trying
        to push back.
        """
        K = next_bank.n_literals // 2
        inc = next_bank.actions()                                   # (Kn, 2K)
        raw = inc[:, :K].astype(np.float32)
        neg = -inc[:, K:].astype(np.float32)
        per_clause_inf = (raw + neg).mean(axis=0)                   # (K,)
        # Scalar magnitude from class credit (since we don't have a class-
        # specific projection for intermediates).
        scalar = float(np.mean(np.abs(per_class_credit)))
        credit = per_clause_inf * scalar
        max_abs = float(np.max(np.abs(credit))) + 1e-9
        return credit / max_abs

    def _trace_decay_step(self) -> None:
        for bank in self.layer_banks:
            bank.trace_step()
        for bank in self.top_banks:
            bank.trace_step()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        shuffle: bool = True,
        verbose: bool = False,
        decay_between_samples: bool = True,
    ) -> "THGTM":
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.int64)
        for epoch in range(epochs):
            order = np.arange(X.shape[0])
            if shuffle:
                self.rng.shuffle(order)
            for i in order:
                self._train_step(X[i], int(y[i]))
                if decay_between_samples:
                    self._trace_decay_step()
            if verbose:
                acc = self.score(X, y)
                print(f"[THGTM] epoch {epoch+1}/{epochs} train_acc={acc:.4f}")
        return self
