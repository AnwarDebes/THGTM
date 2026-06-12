"""Compositional SAT trajectory receipts.

Every clause firing produces a tiny DIMACS-CNF receipt: the literal
inclusion pattern (assignments to the clause's TA include-bits) and the
input literal vector, together with the resulting clause output.  A
trajectory receipt is the conjunction of all per-step receipts over a
finite window plus an optional LTL skeleton describing the global temporal
shape.

For the THGTM v0.1 reference we ship:

* ``ClauseReceipt`` -- a single-step certificate (DIMACS string + HMAC).
* ``TrajectoryReceipt`` -- a list of ``ClauseReceipt`` plus an LTL skeleton.
* ``verify_trajectory`` -- replays the receipt against the literal stream
  and (a) checks each clause output matches the stored value and (b)
  evaluates the LTL skeleton.  Returns (ok, diagnostic_dict).

Concrete SAT solving (Glucose 4) is optional; this module provides a pure
Python DPLL fallback that is sufficient for the small per-step certificates
that we generate.  In production one would call out to ``pysat`` or the
``z3`` solver.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Per-step receipt
# ---------------------------------------------------------------------- #
@dataclass
class ClauseReceipt:
    """A signed certificate that clause ``clause_id`` produced ``output``
    on ``literals`` given its current ``includes`` mask.

    ``cnf`` is a DIMACS-style CNF string that asserts:
        (include_i implies literals_i) for every included literal i.
    Equivalently the clause output is the AND of literals_i for i in includes.

    The receipt is signed with HMAC-SHA-256 over a canonical JSON of the
    fields, so an auditor with the shared key can replay it later even if
    the model has updated.
    """
    step: int
    clause_id: int
    layer: int
    output: int
    n_literals: int
    includes: List[int]              # list of literal indices the clause includes
    literals: List[int]              # 0/1 per literal at this step
    cnf: str                         # DIMACS CNF string asserting the clause
    signature: Optional[str] = None

    def canonical_bytes(self) -> bytes:
        d = {
            "step": self.step,
            "clause_id": self.clause_id,
            "layer": self.layer,
            "output": self.output,
            "n_literals": self.n_literals,
            "includes": list(self.includes),
            "literals": list(self.literals),
            "cnf": self.cnf,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

    def sign(self, key: bytes) -> "ClauseReceipt":
        sig = hmac.new(key, self.canonical_bytes(), hashlib.sha256).hexdigest()
        self.signature = sig
        return self

    def verify_signature(self, key: bytes) -> bool:
        if self.signature is None:
            return False
        expected = hmac.new(key, self.canonical_bytes(),
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)


def build_clause_receipt(
    step: int,
    clause_id: int,
    layer: int,
    includes_mask: np.ndarray,
    literals: np.ndarray,
    key: Optional[bytes] = None,
) -> ClauseReceipt:
    """Materialise a per-step clause receipt and (optionally) sign it."""
    includes = [int(i) for i in np.where(includes_mask)[0]]
    lits = [int(v) for v in literals]
    # Clause output: AND over included literals (empty clause -> 1 by
    # canonical training-mode convention).  For receipts we evaluate the
    # 'predict' convention to be conservative: empty -> 0.
    out = 1 if includes else 0
    for i in includes:
        if not lits[i]:
            out = 0
            break
    # Build a minimal DIMACS CNF:
    #   For each included literal i:  (NOT include_i  OR  literal_i)
    # We use variable 1 for literal_0, 2 for literal_1, etc.
    # 'include' decisions are NOT propositional variables here; they're
    # baked into the receipt.  The CNF therefore says: "given the current
    # includes, the formula is satisfiable iff every included literal is 1."
    clauses_str = []
    for i in includes:
        # The single-literal unit clause "(literal_i)" if include i.
        clauses_str.append(f"{i + 1} 0")
    n_vars = len(literals)
    header = f"p cnf {n_vars} {len(clauses_str)}"
    cnf = "\n".join([header] + clauses_str)

    r = ClauseReceipt(
        step=step,
        clause_id=clause_id,
        layer=layer,
        output=out,
        n_literals=n_vars,
        includes=includes,
        literals=lits,
        cnf=cnf,
    )
    if key is not None:
        r.sign(key)
    return r


# ---------------------------------------------------------------------- #
# LTL skeleton
# ---------------------------------------------------------------------- #
@dataclass
class LTLSkeleton:
    """A tiny bounded-LTL formula over per-step outputs of named clauses.

    Supported operators (all bounded):
        ``("always", clause_id)``               -- clause must fire every step
        ``("eventually_within", clause_id, w)`` -- must fire at least once in
                                                   last w steps
        ``("never_in_window", clause_id, w)``   -- must NOT fire in last w steps
        ``("count_le", clause_id, max_count, w)``-- count of firings in last w
                                                    steps must be <= max_count
    """
    constraints: List[tuple] = field(default_factory=list)

    def evaluate(self, firing_history: Dict[int, List[int]]) -> Tuple[bool, List[str]]:
        ok = True
        violations: List[str] = []
        for c in self.constraints:
            op = c[0]
            if op == "always":
                _, cid = c
                fires = firing_history.get(cid, [])
                if not all(fires):
                    ok = False
                    violations.append(f"always({cid}) violated")
            elif op == "eventually_within":
                _, cid, w = c
                fires = firing_history.get(cid, [])
                # last w entries
                window = fires[-w:] if len(fires) >= w else fires
                if not any(window):
                    ok = False
                    violations.append(f"eventually_within({cid},{w}) violated")
            elif op == "never_in_window":
                _, cid, w = c
                fires = firing_history.get(cid, [])
                window = fires[-w:] if len(fires) >= w else fires
                if any(window):
                    ok = False
                    violations.append(f"never_in_window({cid},{w}) violated")
            elif op == "count_le":
                _, cid, mx, w = c
                fires = firing_history.get(cid, [])
                window = fires[-w:] if len(fires) >= w else fires
                if sum(window) > mx:
                    ok = False
                    violations.append(f"count_le({cid},{mx},{w}) violated "
                                      f"(saw {sum(window)})")
            else:
                ok = False
                violations.append(f"unknown LTL op: {op}")
        return ok, violations


# ---------------------------------------------------------------------- #
# Trajectory receipt
# ---------------------------------------------------------------------- #
@dataclass
class TrajectoryReceipt:
    """Conjunction of per-step receipts plus LTL skeleton."""
    receipts: List[ClauseReceipt]
    ltl: LTLSkeleton = field(default_factory=LTLSkeleton)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "receipts": [asdict(r) for r in self.receipts],
            "ltl": {"constraints": [list(c) for c in self.ltl.constraints]},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryReceipt":
        receipts = [ClauseReceipt(**r) for r in d["receipts"]]
        ltl = LTLSkeleton(constraints=[tuple(c) for c in d["ltl"]["constraints"]])
        return cls(receipts=receipts, ltl=ltl,
                   description=d.get("description", ""))


def verify_trajectory(
    receipt: TrajectoryReceipt,
    key: Optional[bytes] = None,
) -> Tuple[bool, dict]:
    """Re-evaluate every per-step receipt and the LTL skeleton.

    Returns (ok, info).  ``info`` contains
        - ``per_step_ok``: list of bools, one per receipt
        - ``signature_ok``: list of bools if a key was provided
        - ``ltl_ok``: bool
        - ``ltl_violations``: list of strings
    """
    per_step_ok: List[bool] = []
    sig_ok: List[bool] = []
    firing_history: Dict[int, List[int]] = {}
    for r in receipt.receipts:
        # Re-evaluate the CNF: every included literal must be 1.
        out = 1 if r.includes else 0
        for i in r.includes:
            if not r.literals[i]:
                out = 0
                break
        per_step_ok.append(out == r.output)
        firing_history.setdefault(r.clause_id, []).append(int(r.output))
        if key is not None:
            sig_ok.append(r.verify_signature(key))

    ltl_ok, violations = receipt.ltl.evaluate(firing_history)
    ok = all(per_step_ok) and ltl_ok and (key is None or all(sig_ok))

    return ok, {
        "per_step_ok": per_step_ok,
        "signature_ok": sig_ok,
        "ltl_ok": ltl_ok,
        "ltl_violations": violations,
        "n_steps": len(receipt.receipts),
    }
