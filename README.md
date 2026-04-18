# tbgpu

Standalone TinyGPU / NVIDIA runtime extracted from `tinygrad`, without depending on `tinygrad.*` at runtime.

## Scope

- Target platform: macOS + TinyGPU.app + NVIDIA eGPU / tbgpu.
- Execution scope: CUDA Driver-style module load + kernel launch for cubin images.
- Input support: `cuda` and `ptx` example paths are kept by compiling to cubin through `nvcc`.
- Non-goals for this first cut: tinygrad Tensor/op execution, renderer integration, Linux/VFIO paths, NAK/MOCK backends, full CUDA API coverage.

## Layout

- `tbgpu/`: Python package root.
- `tbgpu/cuda_compat.py`: minimal CUDA Driver compatibility layer used by the example.
- `tbgpu/compiler.py`: `nvcc` helpers for `CUDA C -> PTX` and `PTX -> cubin`.
- `tbgpu/nv_backend.py`: thin boundary exposing `open_device` and `load_program`.
- `tbgpu/runtime/transport.py`: TinyGPU remote PCI transport for macOS.
- `tbgpu/runtime/nv/`: vendored low-level NVIDIA bring-up pieces.
- `tbgpu/runtime/device.py`: slim queue, allocator, signal, and device runtime.
- `tbgpu/runtime/program.py`: cubin-only program loader built from the required parts of `NVProgram`.
- `tbgpu/vector_add_demo.py`: demo entry inside the package.

## Dependencies

- Required runtime dependency: `TinyGPU.app` available at `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU`.
- Required for `cuda` / `ptx` example modes: `nvcc` in `PATH`.
- Current low-level bring-up still downloads NVIDIA open-gpu-kernel-modules header / firmware snippets on demand through `runtime/nv/nvdev.py`.

## Run

Use `PYTHONPATH` with the project root that contains the inner `tbgpu/` package:

```bash
PYTHONPATH=$PWD python tests/vector_add.py --kernel-input ptx
PYTHONPATH=$PWD python tests/vector_add.py --kernel-input cuda --launch-mode extra
PYTHONPATH=$PWD python tests/vector_add.py --kernel-input cuda --launch-mode kernel_params
```

## Notes

- The compatibility layer only implements the subset of `cu*` APIs that the example needs.
- The runtime is intentionally organized around device bring-up, memory, program load, and launch, not around tinygrad's original HCQ abstraction.
- A later cleanup step should vendor the NVIDIA snapshot locally instead of fetching it at runtime.
