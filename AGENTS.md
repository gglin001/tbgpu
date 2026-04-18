# Repository Guidelines

## Project Structure & Module Organization

- `tbgpu/` is the main package.
- `tbgpu/runtime/` contains device bring-up, memory management, ELF and program loading, and transport code.
- `tbgpu/support/` holds low-level C helpers.
- `tbgpu/autogen/` contains generated bindings and constants, so avoid routine manual edits there.
- `tests/` stores executable validation scripts, and `tests/kernels/` holds the CUDA and PTX sources they compile or load.
- `scripts/`: scripts for develop.
- `third_party/` is vendored upstream code that should only change during deliberate snapshot updates.
- `debug_agent/`: untracked scratch workspace for temp files and local experiments (use this instead of `/tmp`).

## Build, Test, and Development Commands

- `uv pip install --system --no-build-isolation -e . -v` installs `tbgpu` in editable mode.
- `python tests/vector_add.py --kernel-input ptx` runs the basic smoke test.
- `python tests/matmul.py --verify-suite` and `python tests/flash_attn_v2.py --verify-suite` exercise larger kernels and numeric checks.
- `CUDA_PTI=1 python tests/kernel_profile.py --kernel-input ptx --iters 100` captures event and PTI-like timing data.

Most runtime checks expect `TinyGPU.app` to be available under `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU`. Any `cuda`/`ptx` path also requires `nvcc` in `PATH`.

## Coding Style & Naming Conventions

Use Python with `from __future__ import annotations` and add type hints where they improve clarity. Match the existing 2-space indentation, keep lines within the Ruff limit of 150 characters, and prefer `snake_case` for functions, variables, and modules. Use `UPPER_CASE` for constants such as `KERNEL_NAME` and `MAX_HEAD_DIM`. Run `ruff check .` and `ruff format .` before sending a patch.

## Testing Guidelines

This repository currently relies on script-style integration checks instead of a real `pytest` suite. Add new coverage as a runnable file under `tests/` and place companion kernels in `tests/kernels/`, keeping names aligned, for example `tests/reduce_max.py` with `tests/kernels/reduce_max.cu`. Successful checks should validate numerical results and print the launch shape, timing, or max error.

## Workspace Hygiene and `.gitignore` Policy

The repository uses a narrow `.gitignore` strategy (targeted ignores), not a global deny-all pattern like `*` + whitelist.

- Do not switch to a deny-all ignore pattern unless explicitly requested.
- Assume `.gitignore` controls Git tracking only; it does not block local file reading by agents.
- Treat `third_party/` as a valid source of dependency code and reference implementations when relevant to the task.
- Do not skip `third_party/` or its children merely because they are ignored by `.gitignore`; inspect them when they may affect the work.
- Do not skip `third_party/` or its children merely because they are symlinks; follow and inspect linked contents when relevant and safe.
- Prefer putting disposable outputs in `debug_agent/` instead of expanding broad ignore rules.
