# TP Matlab Toolbox (Python Port)

This folder contains a Python port of the TP Matlab Toolbox 2023 and a runnable implementation of the `UW1` example.

## What's included
- `tptoolbox/`: Python package that mirrors the MATLAB toolbox APIs.
- `examples/uw1.py`: Python port of `UW1.m`.
- `tests/`: Minimal test coverage for core tensor operations.

## Quick start
1. Create/activate a Python environment.
2. Install requirements.
3. Run the UW1 example or tests.

## Notes
This port follows the original toolbox structure and naming conventions where possible. Some algorithms (notably `cno`/`irno`) rely on numerical optimization; results should closely match MATLAB but may differ slightly due to optimizer differences.
