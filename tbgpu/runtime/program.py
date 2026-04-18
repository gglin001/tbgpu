from __future__ import annotations

import re
import struct
import weakref

from tbgpu.autogen import nv_570 as nv_gpu
from tbgpu.helpers import data64_le, hi32, lo32, round_up
from tbgpu.runtime.device import QMD
from tbgpu.runtime.elf import elf_loader


class TBGPUProgram:
  def __init__(self, dev, name: str, cubin: bytes):
    self.dev, self.name, self.lib = dev, name, cubin
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)}
    image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    self.lib_gpu = self.dev.allocator.alloc(round_up((prog_sz := image.nbytes), 0x1000) + 0x1000)
    prog_addr = self.lib_gpu.va_addr
    self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0
    for sh in sections:
      if sh.name == f".nv.shared.{self.name}":
        self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
      if sh.name == f".text.{self.name}":
        prog_addr, prog_sz = self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size
      elif match := re.match(r"\.nv\.constant(\d+)", sh.name):
        self.constbufs[int(match.group(1))] = (self.lib_gpu.va_addr + sh.header.sh_addr, sh.header.sh_size)
      elif sh.name.startswith(".nv.info"):
        for typ, param, data in self._parse_elf_info(sh):
          if sh.name == f".nv.info.{name}" and param == 0xA:
            cbuf0_size = struct.unpack_from("IH", data)[1]
          elif sh.name == ".nv.info" and param == 0x12:
            self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240
          elif sh.name == ".nv.info" and param == 0x2F:
            self.regs_usage = struct.unpack_from("II", data)[1]
    for apply_image_offset, rel_sym_offset, typ, _ in relocs:
      if typ == 2:
        image[apply_image_offset : apply_image_offset + 8] = struct.pack("<Q", self.lib_gpu.va_addr + rel_sym_offset)
      elif typ == 0x38:
        image[apply_image_offset + 4 : apply_image_offset + 8] = struct.pack("<I", (self.lib_gpu.va_addr + rel_sym_offset) & 0xFFFFFFFF)
      elif typ == 0x39:
        image[apply_image_offset + 4 : apply_image_offset + 8] = struct.pack("<I", (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
      else:
        raise RuntimeError(f"unknown NV reloc {typ}")
    min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
    self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()
    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xFFFDC0
      qmd = {
        "qmd_major_version": 5,
        "qmd_type": nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA,
        "program_address_upper_shifted4": hi32(prog_addr >> 4),
        "program_address_lower_shifted4": lo32(prog_addr >> 4),
        "register_count": self.regs_usage,
        "shared_memory_size_shifted7": self.shmem_usage >> 7,
        "shader_local_memory_high_size_shifted4": self.dev.slm_per_thread >> 4,
        "sass_version": dev.sass_version,
      }
    else:
      self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xFFFDC0)]
      qmd = {
        "qmd_major_version": 3,
        "sm_global_caching_enable": 1,
        "program_address_upper": hi32(prog_addr),
        "program_address_lower": lo32(prog_addr),
        "shared_memory_size": self.shmem_usage,
        "register_count_v": self.regs_usage,
        "shader_local_memory_high_size": self.dev.slm_per_thread,
        "sass_version": dev.sass_version,
      }
    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1
    self.qmd = QMD(
      dev,
      **qmd,
      qmd_group_id=0x3F,
      invalidate_texture_header_cache=1,
      invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1,
      invalidate_shader_data_cache=1,
      api_visible_call_limit=1,
      sampler_index=1,
      barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
      constant_buffer_invalidate_0=1,
      min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1A,
      program_prefetch_size=min(prog_sz >> 8, 0x1FF),
      program_prefetch_addr_upper_shifted=prog_addr >> 40,
      program_prefetch_addr_lower_shifted=prog_addr >> 8,
    )
    for index, (addr, size) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(index, addr)
      self.qmd.write(**{f"constant_buffer_size_shifted4_{index}": size, f"constant_buffer_valid_{index}": 1})
    self.kernargs_alloc_size = round_up(self.constbufs[0][1], 1 << 8) + (8 << 8)
    weakref.finalize(self, self.dev.allocator.free, self.lib_gpu)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, size = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off + 4 : start_off + size + 4] if typ == 0x4 else size
      start_off += (size if typ == 0x4 else 0) + 4
