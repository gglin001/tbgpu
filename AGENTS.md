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

The repository uses a narrow `.gitignore` strategy (targeted ignores), not a global deny-all pattern like `*` + whitelist.

- Do not switch to a deny-all ignore pattern unless explicitly requested.
- Assume `.gitignore` controls Git tracking only; it does not block local file reading by agents.
- Treat `third_party/` as a valid source of dependency code and reference implementations when relevant to the task.
- Do not skip `third_party/` or its children merely because they are ignored by `.gitignore`; inspect them when they may affect the work.
- Do not skip `third_party/` or its children merely because they are symlinks; follow and inspect linked contents when relevant and safe.
- Prefer putting disposable outputs in `debug_agent/` instead of expanding broad ignore rules.
