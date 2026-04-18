from __future__ import annotations

from tbgpu.runtime.device import TBGPUDevice
from tbgpu.runtime.program import TBGPUProgram


def open_device(ordinal: int) -> TBGPUDevice:
  return TBGPUDevice(ordinal)


def load_program(device: TBGPUDevice, name: str, image: bytes) -> TBGPUProgram:
  return TBGPUProgram(device, name, image)
