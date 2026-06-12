"""THGTM: Temporal Hierarchical Graph Tsetlin Machine."""

from .etta import EchoTraceAutomaton, ETTABank
from .tm import EttaTsetlinMachine, MultiClassEttaTsetlinMachine, TMConfig
from .hgtm import THGTM, THGTMConfig, LayerConfig
from .temporal import (
    TemporalLiteralEncoder, TemporalOp,
    PAST_k, SINCE, ALWAYS_in_window,
)
from .receipts import (
    ClauseReceipt, TrajectoryReceipt, LTLSkeleton,
    build_clause_receipt, verify_trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "EchoTraceAutomaton",
    "ETTABank",
    "EttaTsetlinMachine",
    "MultiClassEttaTsetlinMachine",
    "TMConfig",
    "THGTM",
    "THGTMConfig",
    "LayerConfig",
    "TemporalLiteralEncoder",
    "TemporalOp",
    "PAST_k",
    "SINCE",
    "ALWAYS_in_window",
    "ClauseReceipt",
    "TrajectoryReceipt",
    "LTLSkeleton",
    "build_clause_receipt",
    "verify_trajectory",
]
