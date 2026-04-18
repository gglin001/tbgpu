from __future__ import annotations

import array
import contextlib
import ctypes
import mmap
from dataclasses import dataclass

from tbgpu.autogen import nv_570 as nv_gpu
from tbgpu.helpers import data64, data64_le, hi32, lo32, round_up
from tbgpu.runtime.common import Buffer, MMIOInterface, Signal
from tbgpu.runtime.memory import AddrSpace, BumpAllocator, VirtMapping
from tbgpu.runtime.nv.nvdev import NVDev
from tbgpu.runtime.transport import APLRemotePCIDevice, System


def nv_flags(reg, **kwargs):
  return ctypes.c_uint32(
    sum(
      ((getattr(nv_gpu, f"{reg}_{key}_{val}".upper()) if isinstance(val, str) else val) << getattr(nv_gpu, f"{reg}_{key}".upper())[1])
      for key, val in kwargs.items()
    )
  ).value


@dataclass(frozen=True)
class PCIAllocationMeta:
  mapping: VirtMapping
  has_cpu_mapping: bool
  hMemory: int = 0


@dataclass
class GPFifo:
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0


class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev: "TBGPUDevice", view: MMIOInterface | None = None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)
    prefix = "NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00"
    if prefix not in QMD.fields:
      QMD.fields[prefix] = {
        **{name[len(prefix) + 1 :]: value for name, value in nv_gpu.__dict__.items() if name.startswith(prefix) and isinstance(value, tuple)},
        **{
          name[len(prefix) + 1 :] + f"_{idx}": value(idx)
          for name, value in nv_gpu.__dict__.items()
          for idx in range(8)
          if name.startswith(prefix) and callable(value)
        },
      }
    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if view is None else view), prefix
    if kwargs:
      self.write(**kwargs)

  def _rw_bits(self, hi: int, lo: int, value: int | None = None):
    mask = ((1 << (width := hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo // 8 : hi // 8 + 1], "little")
    if value is None:
      return (num & mask) >> (lo % 8)
    if value >= (1 << width):
      raise ValueError(f"{value:#x} does not fit")
    self.mv[lo // 8 : hi // 8 + 1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi // 8 - lo // 8 + 1), "little")

  def write(self, **kwargs):
    for key, value in kwargs.items():
      self._rw_bits(*QMD.fields[self.pref][key.upper()], value=value)

  def read(self, key):
    return self._rw_bits(*QMD.fields[self.pref][key.upper()])

  def field_offset(self, key):
    return QMD.fields[self.pref][key.upper()][1] // 8

  def set_constant_buf_addr(self, index: int, addr: int):
    if self.ver < 4:
      self.write(**{f"constant_buffer_addr_upper_{index}": hi32(addr), f"constant_buffer_addr_lower_{index}": lo32(addr)})
    else:
      self.write(**{f"constant_buffer_addr_upper_shifted6_{index}": hi32(addr >> 6), f"constant_buffer_addr_lower_shifted6_{index}": lo32(addr >> 6)})


class NVCommandQueue:
  def __init__(self):
    self._q: list[int] = []
    self.active_qmd = None
    self.active_qmd_buf = None

  def q(self, *values: int):
    self._q.extend(values)
    return self

  def nvm(self, subchannel: int, mthd: int, *args: int, typ=2):
    return self.q((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args)

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None, local_mem=None, local_mem_tpc_bytes=None):
    if compute_class is not None:
      self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class is not None:
      self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window is not None:
      self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window is not None:
      self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem is not None:
      self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes is not None:
      self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xFF)
    return self

  def wait(self, signal: Signal, value: int = 0):
    self.nvm(
      0,
      nv_gpu.NVC56F_SEM_ADDR_LO,
      *data64_le(signal.value_addr),
      *data64_le(value),
      nv_flags("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"),
    )
    self.active_qmd = None
    return self

  def _submit_to_gpfifo(self, dev: "TBGPUDevice", gpfifo: GPFifo):
    cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
    cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
    dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array("I", self._q)
    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr // 4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count
    System.memory_barrier()
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1
    return self

  def submit(self, dev: "TBGPUDevice"):
    self._submit(dev)
    return self


def _write_values(mem: MMIOInterface, fmt: str, offset: int, *values: int, mask: int | None = None):
  view = mem.view(offset=offset, size=len(values) * ctypes.sizeof({"I": ctypes.c_uint32, "H": ctypes.c_uint16, "B": ctypes.c_uint8}[fmt]), fmt=fmt)
  for index, value in enumerate(values):
    view[index] = value if mask is None else ((view[index] & ~mask) | value)


class NVComputeQueue(NVCommandQueue):
  def memory_barrier(self):
    self.nvm(
      1,
      nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
      nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"),
    )
    self.active_qmd = None
    return self

  def exec(self, prg, args_buf: Buffer, global_size: tuple[int, int, int], local_size: tuple[int, int, int]):
    qmd_buf = args_buf.offset(round_up(prg.constbufs[0][1], 1 << 8))
    qmd_buf.cpu_view().view(size=prg.qmd.mv.nbytes, fmt="B")[:] = prg.qmd.mv
    qmd = QMD(dev=prg.dev, view=qmd_buf.cpu_view())
    _write_values(qmd_buf.cpu_view(), "I", qmd.field_offset("cta_raster_width" if qmd.ver < 4 else "grid_width"), *global_size)
    _write_values(qmd_buf.cpu_view(), "H", qmd.field_offset("cta_thread_dimension0"), *local_size[:2])
    _write_values(qmd_buf.cpu_view(), "B", qmd.field_offset("cta_thread_dimension2"), local_size[2])
    qmd.set_constant_buf_addr(0, args_buf.va_addr)
    if self.active_qmd is None:
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  def signal(self, signal: Signal, value: int = 0):
    if self.active_qmd is not None:
      for index in range(2):
        if self.active_qmd.read(f"release{index}_enable") == 0:
          self.active_qmd.write(**{f"release{index}_enable": 1})
          addr_off = self.active_qmd.field_offset(
            f"release{index}_address_lower" if self.active_qmd.ver < 4 else f"release_semaphore{index}_addr_lower"
          )
          _write_values(self.active_qmd_buf.cpu_view(), "I", addr_off, signal.value_addr & 0xFFFFFFFF)
          _write_values(self.active_qmd_buf.cpu_view(), "I", addr_off + 4, signal.value_addr >> 32, mask=0xF)
          val_off = self.active_qmd.field_offset(
            f"release{index}_payload_lower" if self.active_qmd.ver < 4 else f"release_semaphore{index}_payload_lower"
          )
          _write_values(self.active_qmd_buf.cpu_view(), "I", val_off, value & 0xFFFFFFFF)
          _write_values(self.active_qmd_buf.cpu_view(), "I", val_off + 4, value >> 32)
          return self
    self.nvm(
      0,
      nv_gpu.NVC56F_SEM_ADDR_LO,
      *data64_le(signal.value_addr),
      *data64_le(value),
      nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en"),
    )
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit(self, dev: "TBGPUDevice"):
    self._submit_to_gpfifo(dev, dev.compute_gpfifo)


class NVCopyQueue(NVCommandQueue):
  def copy(self, dest: Buffer, src: Buffer, copy_size: int):
    for off in range(0, copy_size, 1 << 31):
      chunk = min(copy_size - off, 1 << 31)
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src.va_addr + off), *data64(dest.va_addr + off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, chunk)
      self.nvm(
        4,
        nv_gpu.NVC6B5_LAUNCH_DMA,
        nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"),
      )
    return self

  def signal(self, signal: Signal, value: int = 0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore"))
    return self

  def _submit(self, dev: "TBGPUDevice"):
    self._submit_to_gpfifo(dev, dev.dma_gpfifo)


class PCIIface:
  def __init__(self, dev: "TBGPUDevice", dev_id: int):
    self.pci_dev = APLRemotePCIDevice("NV", dev_id)
    self.dev_impl = NVDev(self.pci_dev)
    self.dev, self.vram_bar = dev, 1
    with contextlib.suppress(Exception):
      self.pci_dev.resize_bar(self.vram_bar)
    self.root, self.gpu_instance = 0xC1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())
    gsp = self.dev_impl.gsp
    self.gpfifo_class, self.compute_class, self.dma_class = gsp.gpfifo_class, gsp.compute_class, gsp.dma_class
    self.viddec_class = None

  def is_bar_small(self) -> bool:
    return self.pci_dev.bar_info(self.vram_bar)[1] == (256 << 20)

  def setup_usermode(self):
    return 0xCE000000, self.pci_dev.map_bar(bar=0, fmt="I", off=0xBB0000, size=0x10000)

  def setup_vm(self, vaspace):
    return None

  def setup_gpfifo_vm(self, gpfifo):
    return None

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)

  def rm_control(self, obj, cmd, params=None, **kwargs):
    return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root, **kwargs)

  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp():
      pass
    if self.dev_impl.is_err_state:
      raise RuntimeError("device fault detected")

  def device_fini(self):
    self.dev_impl.fini()

  def alloc(self, size: int, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False):
    should_use_sysmem = host or ((cpu_access if self.is_bar_small() else (uncached and cpu_access)) and not force_devmem)
    size = round_up(size, mmap.PAGESIZE if should_use_sysmem else ((2 << 20) if size >= (8 << 20) else (4 << 10)))
    if should_use_sysmem:
      vaddr = self.dev_impl.mm.alloc_vaddr(size, align=mmap.PAGESIZE)
      memview, paddrs = self.pci_dev.alloc_sysmem(size, vaddr=vaddr, contiguous=contiguous)
      mapping = self.dev_impl.mm.map_range(vaddr, size, [(paddr, 0x1000) for paddr in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
      return Buffer(vaddr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=True, hMemory=paddrs[0]), view=memview, owner=self.dev)
    mapping = self.dev_impl.mm.valloc(size, uncached=uncached, contiguous=cpu_access)
    barview = self.pci_dev.map_bar(bar=self.vram_bar, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return Buffer(mapping.va_addr, size, meta=PCIAllocationMeta(mapping, cpu_access, hMemory=mapping.paddrs[0][0]), view=barview, owner=self.dev)

  def free(self, buf: Buffer):
    mapping = buf.meta.mapping
    if mapping.aspace is AddrSpace.PHYS:
      self.dev_impl.mm.vfree(mapping)
    else:
      self.dev_impl.mm.unmap_range(buf.va_addr, buf.size)


class TBGPUAllocator:
  def __init__(self, dev: "TBGPUDevice", staging_size=(2 << 20), staging_count=2):
    self.dev = dev
    self.staging = [self.dev.iface.alloc(staging_size, host=True, cpu_access=True) for _ in range(staging_count)]
    self.staging_idx = 0
    self.staging_timeline = [0] * staging_count

  def alloc(self, size: int, *, cpu_access=False, host=False, uncached=False, contiguous=False, force_devmem=False) -> Buffer:
    return self.dev.iface.alloc(size, cpu_access=cpu_access, host=host, uncached=uncached, contiguous=contiguous, force_devmem=force_devmem)

  def free(self, buf: Buffer, size: int | None = None):
    self.dev.iface.free(buf.base)

  def _next_staging(self) -> Buffer:
    self.staging_idx = (self.staging_idx + 1) % len(self.staging)
    self.dev.timeline_signal.wait(self.staging_timeline[self.staging_idx])
    return self.staging[self.staging_idx]

  def _copyin(self, dest: Buffer, src: memoryview):
    for off in range(0, src.nbytes, self.staging[0].size):
      stage = self._next_staging()
      chunk = min(stage.size, src.nbytes - off)
      stage.cpu_view().view(size=chunk, fmt="B")[:] = src.cast("B")[off : off + chunk]
      NVCopyQueue().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).copy(dest.offset(off, chunk), stage.offset(0, chunk), chunk).signal(
        self.dev.timeline_signal, self.dev.next_timeline()
      ).submit(self.dev)
      self.staging_timeline[self.staging_idx] = self.dev.timeline_value - 1

  def _copyout(self, dest: memoryview, src: Buffer):
    for off in range(0, dest.nbytes, self.staging[0].size):
      stage = self._next_staging()
      chunk = min(stage.size, dest.nbytes - off)
      NVCopyQueue().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).copy(stage.offset(0, chunk), src.offset(off, chunk), chunk).signal(
        self.dev.timeline_signal, self.dev.next_timeline()
      ).submit(self.dev)
      self.staging_timeline[self.staging_idx] = self.dev.timeline_value - 1
      self.dev.synchronize()
      dest.cast("B")[off : off + chunk] = stage.cpu_view().view(size=chunk, fmt="B")[:]

  def _transfer(self, dest: Buffer, src: Buffer, size: int, src_dev: "TBGPUDevice", dest_dev: "TBGPUDevice"):
    if src_dev is not dest_dev:
      raise NotImplementedError("cross-device transfers are not supported in this minimal runtime")
    NVCopyQueue().wait(src_dev.timeline_signal, src_dev.timeline_value - 1).copy(dest, src, size).signal(
      src_dev.timeline_signal, src_dev.next_timeline()
    ).submit(src_dev)


class TBGPUDevice:
  def __init__(self, ordinal: int = 0):
    self.device_id = ordinal
    self.iface = PCIIface(self, ordinal)
    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(
      deviceId=self.iface.gpu_instance, hClientShare=self.iface.root, vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES
    )
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1FFFFFFFFFFFF))
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()
    self.iface.rm_control(
      self.subdevice,
      nv_gpu.NV2080_CTRL_CMD_PERF_BOOST,
      nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(
        duration=0xFFFFFFFF,
        flags=(
          (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4)
          | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6)
          | nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX
        ),
      ),
    )
    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(
      vaBase=0x1000,
      vaSize=0x1FFFFFB000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
    )
    vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)
    self.iface.setup_vm(vaspace)
    self.channel_group = self.iface.rm_alloc(
      self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    )
    self.gpfifo_area = self.iface.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True, uncached=False)
    ctxshare = self.iface.rm_alloc(
      self.channel_group,
      nv_gpu.FERMI_CONTEXT_SHARE_A,
      nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC),
    )
    self.compute_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0, entries=0x10000, compute=True)
    self.dma_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    self.cmdq_page = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt="I")
    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info(
      "num_gpcs", "num_tpc_per_gpc", "num_sm_per_tpc", "max_warps_per_sm", "sm_version"
    )
    self.arch = (
      "sm_120" if self.sm_version == 0xA04 else f"sm_{(self.sm_version >> 8) & 0xFF}{(val >> 4) if (val := self.sm_version & 0xFF) > 0xF else val}"
    )
    self.sass_version = ((self.sm_version & 0xF00) >> 4) | (self.sm_version & 0xF)
    self.allocator = TBGPUAllocator(self)
    self._signal_pages: list[Buffer] = []
    self._signal_alloc: BumpAllocator | None = None
    self.timeline_signal = self.new_signal()
    self.timeline_value = 1
    self.kernargs_buf = self.iface.alloc(16 << 20, cpu_access=True)
    self.kernargs_offset_allocator = BumpAllocator(self.kernargs_buf.size, wrap=True)
    self._setup_gpfifos()

  def is_nvd(self) -> bool:
    return isinstance(self.iface, PCIIface)

  def new_signal(self) -> Signal:
    if self._signal_alloc is None or self._signal_alloc.ptr + 16 > self._signal_alloc.size:
      page = self.iface.alloc(0x1000, host=True, uncached=True, cpu_access=True)
      self._signal_pages.append(page)
      self._signal_alloc = BumpAllocator(page.size, base=page.va_addr, wrap=False)
    off = self._signal_alloc.alloc(16, 16)
    page = self._signal_pages[-1]
    return Signal(page.offset(off - page.va_addr, 16), owner=self)

  def synchronize(self, timeout: int | None = None):
    self.timeline_signal.wait(self.timeline_value - 1, timeout=timeout)

  def next_timeline(self) -> int:
    self.timeline_value += 1
    return self.timeline_value - 1

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False):
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
      gpFifoOffset=gpfifo_area.va_addr + offset,
      gpFifoEntries=entries,
      hContextShare=ctxshare,
      hObjectError=notifier.meta.hMemory,
      hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32 * 8)(gpfifo_area.meta.hMemory),
      userdOffset=(ctypes.c_uint64 * 8)(entries * 8 + offset),
      engineType=19 if video else 0,
    )
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)
    if compute:
      self.debug_compute_obj, self.debug_channel = self.iface.rm_alloc(gpfifo, self.iface.compute_class), gpfifo
      self.debugger = self.iface.rm_alloc(
        self.nvdevice, nv_gpu.GT200_DEBUGGER, nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=self.iface.root, hClass3dObject=self.debug_compute_obj)
      )
    else:
      self.iface.rm_alloc(gpfifo, self.iface.dma_class)
    token = self.iface.rm_control(
      gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    ).workSubmitToken
    return GPFifo(
      ring=gpfifo_area.cpu_view().view(offset, entries * 8, fmt="Q"),
      entries_count=entries,
      token=token,
      gpput=gpfifo_area.cpu_view().view(offset + entries * 8 + getattr(nv_gpu.AmpereAControlGPFifo, "GPPut").offset, fmt="I"),
    )

  def _query_gpu_info(self, *reqs):
    nvrs = [
      getattr(nv_gpu, "NV2080_CTRL_GR_INFO_INDEX_" + req.upper(), getattr(nv_gpu, "NV2080_CTRL_GR_INFO_INDEX_LITTER_" + req.upper(), None))
      for req in reqs
    ]
    if self.is_nvd():
      info = self.iface.rm_control(
        self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO, nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS()
      )
      return [info.engineInfo[0].infoList[nvr].data for nvr in nvrs]
    infos = (nv_gpu.NV2080_CTRL_GR_INFO * len(nvrs))(*[nv_gpu.NV2080_CTRL_GR_INFO(index=nvr) for nvr in nvrs])
    self.iface.rm_control(
      self.subdevice,
      nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO,
      nv_gpu.NV2080_CTRL_GR_GET_INFO_PARAMS(grInfoListSize=len(infos), grInfoList=ctypes.addressof(infos)),
    )
    return [item.data for item in infos]

  def _setup_gpfifos(self):
    self.slm_per_thread, self.shader_local_mem = 0, None
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
    NVComputeQueue().setup(
      compute_class=self.iface.compute_class, local_mem_window=self.local_mem_window, shared_mem_window=self.shared_mem_window
    ).signal(self.timeline_signal, self.next_timeline()).submit(self)
    NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1).setup(copy_class=self.iface.dma_class).signal(
      self.timeline_signal, self.next_timeline()
    ).submit(self)
    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required:
      return
    self.slm_per_thread = round_up(required, 32)
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    if self.shader_local_mem is not None:
      self.allocator.free(self.shader_local_mem)
    self.shader_local_mem = self.allocator.alloc(round_up(bytes_per_tpc * self.num_tpc_per_gpc * self.num_gpcs, 0x20000))
    NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1).setup(
      local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc
    ).signal(self.timeline_signal, self.next_timeline()).submit(self)
