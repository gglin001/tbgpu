from __future__ import annotations

import ctypes

from tbgpu.support import c

dll = c.DLL("libc", "c", use_errno=True)


@dll.bind(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64)
def mmap(__addr: ctypes.c_void_p, __len: int, __prot: int, __flags: int, __fd: int, __offset: int) -> ctypes.c_void_p: ...


@dll.bind(ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t)
def munmap(__addr: ctypes.c_void_p, __len: int) -> int: ...


@dll.bind(ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t)
def mlock(__addr: ctypes.c_void_p, __len: int) -> int: ...


@dll.bind(ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)
def madvise(__addr: ctypes.c_void_p, __len: int, __advice: int) -> int: ...


SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_RELA = 4
SHT_REL = 9


def ELF64_R_SYM(info: int) -> int:
  return info >> 32


def ELF64_R_TYPE(info: int) -> int:
  return info & 0xFFFFFFFF


@c.record
class Elf64_Ehdr(c.Struct):
  SIZE = 64
  e_ident: ctypes.Array
  e_type: int
  e_machine: int
  e_version: int
  e_entry: int
  e_phoff: int
  e_shoff: int
  e_flags: int
  e_ehsize: int
  e_phentsize: int
  e_phnum: int
  e_shentsize: int
  e_shnum: int
  e_shstrndx: int


Elf64_Ehdr.register_fields([
  ("e_ident", ctypes.c_ubyte * 16, 0),
  ("e_type", ctypes.c_uint16, 16),
  ("e_machine", ctypes.c_uint16, 18),
  ("e_version", ctypes.c_uint32, 20),
  ("e_entry", ctypes.c_uint64, 24),
  ("e_phoff", ctypes.c_uint64, 32),
  ("e_shoff", ctypes.c_uint64, 40),
  ("e_flags", ctypes.c_uint32, 48),
  ("e_ehsize", ctypes.c_uint16, 52),
  ("e_phentsize", ctypes.c_uint16, 54),
  ("e_phnum", ctypes.c_uint16, 56),
  ("e_shentsize", ctypes.c_uint16, 58),
  ("e_shnum", ctypes.c_uint16, 60),
  ("e_shstrndx", ctypes.c_uint16, 62),
])


@c.record
class Elf64_Shdr(c.Struct):
  SIZE = 64
  sh_name: int
  sh_type: int
  sh_flags: int
  sh_addr: int
  sh_offset: int
  sh_size: int
  sh_link: int
  sh_info: int
  sh_addralign: int
  sh_entsize: int


Elf64_Shdr.register_fields([
  ("sh_name", ctypes.c_uint32, 0),
  ("sh_type", ctypes.c_uint32, 4),
  ("sh_flags", ctypes.c_uint64, 8),
  ("sh_addr", ctypes.c_uint64, 16),
  ("sh_offset", ctypes.c_uint64, 24),
  ("sh_size", ctypes.c_uint64, 32),
  ("sh_link", ctypes.c_uint32, 40),
  ("sh_info", ctypes.c_uint32, 44),
  ("sh_addralign", ctypes.c_uint64, 48),
  ("sh_entsize", ctypes.c_uint64, 56),
])


@c.record
class Elf64_Sym(c.Struct):
  SIZE = 24
  st_name: int
  st_info: int
  st_other: int
  st_shndx: int
  st_value: int
  st_size: int


Elf64_Sym.register_fields([
  ("st_name", ctypes.c_uint32, 0),
  ("st_info", ctypes.c_uint8, 4),
  ("st_other", ctypes.c_uint8, 5),
  ("st_shndx", ctypes.c_uint16, 6),
  ("st_value", ctypes.c_uint64, 8),
  ("st_size", ctypes.c_uint64, 16),
])


@c.record
class Elf64_Rel(c.Struct):
  SIZE = 16
  r_offset: int
  r_info: int


Elf64_Rel.register_fields([
  ("r_offset", ctypes.c_uint64, 0),
  ("r_info", ctypes.c_uint64, 8),
])


@c.record
class Elf64_Rela(c.Struct):
  SIZE = 24
  r_offset: int
  r_info: int
  r_addend: int


Elf64_Rela.register_fields([
  ("r_offset", ctypes.c_uint64, 0),
  ("r_info", ctypes.c_uint64, 8),
  ("r_addend", ctypes.c_int64, 16),
])
