# Repository Guidelines

## Project Structure & Module Organization

- `tbgpu/` is the main Python package. Core runtime code lives in `tbgpu/runtime/`, NVIDIA-specific pieces in `tbgpu/runtime/nv/`, low-level helpers in `tbgpu/support/`, and generated bindings in `tbgpu/autogen/`.
- `tbgpu/tinygpu/` is a separate macOS/Xcode app and driver tree. Keep edits there intentional, and avoid unrelated project-file churn.
- `tests/` contains script-style integration checks, with kernels in `tests/kernels/`.
- `scripts/`: scripts for develop.
- `third_party/` is vendored upstream code that should only change during deliberate snapshot updates.
- `debug_agent/`: untracked scratch workspace for temp files and local experiments (use this instead of `/tmp`).

## Build, Test, and Development Commands

- `uv pip install --system --no-build-isolation -e . -v` installs `tbgpu` in editable mode.
- `python tests/vector_add.py --kernel-input ptx` runs the basic smoke test.
- `python tests/matmul.py --verify-suite` and `python tests/flash_attn_v2.py --verify-suite` exercise larger kernels and numeric checks.
- `CUDA_PTI=1 python tests/kernel_profile.py --kernel-input ptx --iters 100` captures event and PTI-like timing data.
- Most runtime checks expect `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU`. CUDA and PTX flows also need `nvcc` in `PATH`.

## Coding Style & Naming Conventions

- Use Python with `from __future__ import annotations` in new modules, 2-space indentation, type hints where they help, `snake_case` names, and `UPPER_CASE` constants.
- Run `ruff check .` and `ruff format .`. For larger changes, also run `pre-commit run --all-files`.
- Do not hand-edit `tbgpu/autogen/` unless you are intentionally regenerating bindings.

## Workspace Hygiene and `.gitignore` Policy

- Keep `.gitignore` narrow and targeted; do not switch to a deny-all whitelist pattern unless explicitly requested.
- `.gitignore` only affects Git tracking, so agents may still read ignored files, including relevant code under `third_party/` and safe symlinked contents.
- When searching under `third_party/`, prefer `rg -u` or `rg -uL` so `.gitignore` rules and symlinks do not hide relevant files.
- Put disposable scripts and outputs in `debug_agent/` instead of broadening ignore rules.

## Agent Scratch Workflow

- For debugging, repro, validation, or inspection, prefer saving helper scripts, fixtures, and outputs under `debug_agent/` and running them from there.
- Use descriptive names such as `debug_agent/repro_matmul_stride.py`, and keep useful scratch artifacts during the task so the workflow stays visible and reproducible.
- `python - <<'PY'` is a discouraged style example; reserve inline heredocs or one-liners for truly tiny throwaway commands, and otherwise default to saved files in `debug_agent/`.
