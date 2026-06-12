# Changelog

All notable changes to THGTM are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-12

Reference implementation accompanying the paper.

### Added

- Echo-Trace Tsetlin Automaton (ETTA) bank: one float of decaying trace
  per automaton, enabling temporal credit assignment.
- Binary and multi-class TM trainers on the ETTA substrate; lambda = 0
  reduces bit-exactly to the vanilla TM.
- Stacked GraphTM layers with trace-projected feedback for multi-layer
  credit assignment.
- Bounded-LTL temporal literals: PAST_k, SINCE, ALWAYS_in_window.
- DIMACS CNF + HMAC clause receipts with trajectory-level composition.
- Four reproducible experiments (noisy XOR sanity, temporal XOR,
  depth-N parity, slow-roll exfiltration trajectory verification),
  under 5 minutes total on a single CPU.
- Figure-generation script, LaTeX paper, 25 unit tests.

### Known limitations (v0.1)

- No convergence proof for trace-projected feedback.
- Modest multi-layer uplift beyond path length 2 on depth-N parity.
- Trajectory benchmark is synthetic and intentionally simple.

[0.1.0]: https://github.com/AnwarDebes/THGTM/releases/tag/v0.1.0
