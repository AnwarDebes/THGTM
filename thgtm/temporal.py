"""Bounded-LTL temporal literals for THGTM.

A small library of temporal-feature encoders that turn a stream of
Boolean inputs into a Boolean literal vector that an ETTA-based TM can
consume directly.  Three operators are provided:

* ``PAST_k(literal, k)``   -- value of ``literal`` exactly ``k`` steps ago.
* ``SINCE(a, b, T)``       -- 1 iff ``a`` was true at some t' in [t-T, t] and
                              ``b`` has been true at every step in (t', t].
* ``ALWAYS_in_window(c, w)`` -- 1 iff ``c`` was true at every step in
                                [t-w+1, t].

The encoder maintains a small ring buffer of the last ``max_history``
input vectors so that any combination of these operators can be evaluated
in O(history * n_inputs) per step.  No backprop, no autograd -- this is a
deterministic preprocessing layer for the TM.

The encoder is intentionally *separate* from the ETTA trace.  The trace
modulates *feedback timing* during learning; the temporal literals modulate
*what the clauses see at inference*.  Together they make THGTM a true
temporal learner without any recurrence in the classical NN sense.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class TemporalOp:
    """A single temporal feature recipe.  ``op`` is one of
    ``'past'``, ``'since'``, ``'always_in_window'``.  ``args`` is the tuple
    of integer arguments specific to the op."""
    op: str
    args: tuple


def PAST_k(input_index: int, k: int) -> TemporalOp:
    if k < 1:
        raise ValueError("PAST_k requires k >= 1")
    return TemporalOp("past", (int(input_index), int(k)))


def SINCE(a_index: int, b_index: int, T: int) -> TemporalOp:
    if T < 1:
        raise ValueError("SINCE requires T >= 1")
    return TemporalOp("since", (int(a_index), int(b_index), int(T)))


def ALWAYS_in_window(c_index: int, w: int) -> TemporalOp:
    if w < 1:
        raise ValueError("ALWAYS_in_window requires w >= 1")
    return TemporalOp("always_in_window", (int(c_index), int(w)))


class TemporalLiteralEncoder:
    """Stateful encoder that augments each Boolean input with temporal features.

    Usage::

        enc = TemporalLiteralEncoder(n_inputs=2, ops=[
            PAST_k(0, 1), PAST_k(0, 2), SINCE(0, 1, 4),
        ])
        for x_t in stream:
            x_aug = enc.transform(x_t)   # length n_inputs + len(ops)
            ...

    ``transform`` updates the internal buffer first and then computes the
    operator values, so ``PAST_k`` at the very first step reads zero (it
    has no history yet).
    """

    def __init__(self, n_inputs: int, ops: Sequence[TemporalOp]):
        self.n_inputs = int(n_inputs)
        self.ops: List[TemporalOp] = list(ops)
        # We need history at most max_lookback steps.
        self.max_lookback = max(
            (op.args[1] if op.op == "past"
             else op.args[2] if op.op == "since"
             else op.args[1])
            for op in self.ops
        ) if self.ops else 0
        self.buffer: deque[np.ndarray] = deque(maxlen=self.max_lookback + 1)
        self.n_outputs = self.n_inputs + len(self.ops)

    def reset(self) -> None:
        self.buffer.clear()

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int8)
        if x.shape != (self.n_inputs,):
            raise ValueError(f"x must have shape ({self.n_inputs},)")
        self.buffer.append(x.copy())
        out = np.zeros(self.n_outputs, dtype=np.int8)
        out[: self.n_inputs] = x
        # buffer[-1] is current, buffer[-2] is t-1, etc.
        n_hist = len(self.buffer)
        for j, op in enumerate(self.ops):
            out_idx = self.n_inputs + j
            if op.op == "past":
                i, k = op.args
                if n_hist > k:
                    out[out_idx] = self.buffer[-1 - k][i]
                else:
                    out[out_idx] = 0
            elif op.op == "since":
                a, b, T = op.args
                # Look backward for the latest t' in [t-T, t] where a was true,
                # then check b has been true on every step (t', t].
                val = 0
                upper = min(T, n_hist - 1)
                for offset in range(0, upper + 1):
                    if self.buffer[-1 - offset][a]:
                        # check b on steps -offset+1 .. 0  (exclusive of t')
                        if offset == 0:
                            val = 1  # a true at t -> trivially "since"
                            break
                        sub = list(self.buffer)[-offset:]   # offset items, inclusive of t
                        if all(s[b] for s in sub):
                            val = 1
                        break
                out[out_idx] = val
            elif op.op == "always_in_window":
                c, w = op.args
                w_eff = min(w, n_hist)
                sub = list(self.buffer)[-w_eff:]
                out[out_idx] = 1 if all(s[c] for s in sub) else 0
            else:
                raise ValueError(f"Unknown temporal op: {op.op}")
        return out

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Transform a whole stream in order (does NOT reset)."""
        X = np.asarray(X, dtype=np.int8)
        out = np.zeros((X.shape[0], self.n_outputs), dtype=np.int8)
        for i in range(X.shape[0]):
            out[i] = self.transform(X[i])
        return out
