"""In-place ELF64 dynamic symbol version downgrader for Rocky Linux 9.8 (`GLIBC <= 2.34`) compatibility.

When native C/C++ extensions (`PyAlembic`, `PyImath`, `fbxio`) are compiled on a newer glibc host
(e.g. glibc 2.38–2.43), GCC/glibc headers redirect standard C89 math and integer parsing symbols:
  - `__isoc23_strtol@GLIBC_2.38` (ABI-identical to `strtol@GLIBC_2.2.5`)
  - `acosf`, `asinf`, `atan2f`, `sqrtf`, `coshf`, `sinhf`, `log10f` `@GLIBC_2.43` (ABI-identical to `@GLIBC_2.2.5`)
  - `fmod`, `fmodf` `@GLIBC_2.38` (ABI-identical to `@GLIBC_2.2.5`)

On Rocky Linux 9.8 (`glibc 2.34`), `dlopen` fails with `version 'GLIBC_2.38' not found` unless those
undefined symbol references and `.gnu.version_r` (`SHT_GNU_verneed`) entries are bound to `GLIBC_2.2.5`.

This pure-Python healer patches ELF64 binaries in-place without changing file size, section offsets,
or program headers, and runs idempotently in < 30 ms during `relocate-venv` / startup.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

SHT_DYNSYM = 11
SHT_GNU_verneed = 0x6FFFFFFE
SHT_GNU_versym = 0x6FFFFFFF

TARGET_LIBS = {b"libc.so.6", b"libm.so.6", b"libpthread.so.0", b"libdl.so.2", b"librt.so.1"}
ISOC23_PREFIX = b"__isoc23_"


def _elf_hash(name: bytes) -> int:
    h = 0
    for b in name:
        h = ((h << 4) + b) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def _read_cstr(buf: bytes | bytearray, off: int) -> bytes:
    end = buf.find(b"\x00", off)
    return bytes(buf[off:end]) if end >= 0 else bytes(buf[off:])


def _parse_glibc_ver(ver: bytes) -> tuple[int, ...] | None:
    if not ver.startswith(b"GLIBC_"):
        return None
    try:
        return tuple(int(x) for x in ver[6:].decode("ascii").split("."))
    except ValueError:
        return None


def heal_elf_glibc_compat(path: str | Path, max_glibc: tuple[int, ...] = (2, 34)) -> bool:
    """Downgrade `GLIBC_2.35+` undefined symbol bindings (`__isoc23_strto*` and C89 math) to `GLIBC_2.2.5`.
    Returns `True` if the file was modified, `False` if already compatible or not an ELF64 binary."""
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        return False
    try:
        with p.open("rb") as f:
            head = f.read(64)
            if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
                return False
            data = bytearray(head + f.read())
    except OSError:
        return False

    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize, e_shnum, _ = struct.unpack_from("<HHH", data, 58)
    if e_shoff == 0 or e_shnum == 0 or e_shentsize < 64 or e_shoff + e_shentsize * e_shnum > len(data):
        return False

    sections = []
    dynsym_sh = None
    versym_sh = None
    verneed_sh = None

    for idx in range(e_shnum):
        off = e_shoff + idx * e_shentsize
        sh = struct.unpack_from("<IIQQQQIIQQ", data, off)
        sections.append((off, sh))
        sh_type = sh[1]
        if sh_type == SHT_DYNSYM:
            dynsym_sh = (idx, off, sh)
        elif sh_type == SHT_GNU_versym:
            versym_sh = (idx, off, sh)
        elif sh_type == SHT_GNU_verneed:
            verneed_sh = (idx, off, sh)

    if dynsym_sh is None or versym_sh is None or verneed_sh is None:
        return False

    dynstr_idx = verneed_sh[2][6]  # sh_link -> .dynstr
    if dynstr_idx >= len(sections):
        return False
    dynstr_sh = sections[dynstr_idx][1]
    dynstr_off, dynstr_size = dynstr_sh[4], dynstr_sh[5]
    dynstr = bytes(data[dynstr_off : dynstr_off + dynstr_size])

    vn_sec_off = verneed_sh[2][4]
    vn_sec_size = verneed_sh[2][5]
    vn_num = verneed_sh[2][7]  # sh_info = count of Elf64_Verneed entries

    # Quick scan: check if any Vernaux entry has GLIBC > max_glibc
    remap_ver_idx: dict[int, int] = {}
    changed = False

    # Find an existing "GLIBC_2.2.5\0" offset in .dynstr as fallback if a Verneed lacks GLIBC_2.2.5
    glibc_225_str_off = dynstr.find(b"GLIBC_2.2.5\x00")

    vn_pos = vn_sec_off
    for _ in range(vn_num):
        if vn_pos + 16 > vn_sec_off + vn_sec_size:
            break
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, vn_pos)
        file_name = _read_cstr(dynstr, vn_file)

        if file_name in TARGET_LIBS and vn_cnt > 0:
            aux_list = []
            aux_pos = vn_pos + vn_aux
            for _k in range(vn_cnt):
                if aux_pos + 16 > vn_sec_off + vn_sec_size:
                    break
                vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aux_pos)
                ver_name = _read_cstr(dynstr, vna_name)
                aux_list.append([aux_pos, vna_hash, vna_flags, vna_other, vna_name, vna_next, ver_name])
                if vna_next == 0:
                    break
                aux_pos += vna_next

            high_entries = []
            low_entries = []
            base_225_other: int | None = None
            lowest_glibc: tuple[tuple[int, ...], int] | None = None

            for entry in aux_list:
                gv = _parse_glibc_ver(entry[6])
                if gv is not None and gv > max_glibc:
                    high_entries.append(entry)
                else:
                    low_entries.append(entry)
                    if entry[6] == b"GLIBC_2.2.5":
                        base_225_other = entry[3] & 0x7FFF
                    if gv is not None and (lowest_glibc is None or gv < lowest_glibc[0]):
                        lowest_glibc = (gv, entry[3] & 0x7FFF)

            if high_entries:
                changed = True
                if base_225_other is not None:
                    target_other = base_225_other
                elif lowest_glibc is not None:
                    target_other = lowest_glibc[1]
                elif glibc_225_str_off >= 0:
                    # Convert the first high Vernaux in-place into GLIBC_2.2.5 so Verneed stays non-empty!
                    keep_e = high_entries.pop(0)
                    keep_e[1] = _elf_hash(b"GLIBC_2.2.5")
                    keep_e[2] = 0
                    keep_e[4] = glibc_225_str_off
                    keep_e[6] = b"GLIBC_2.2.5"
                    low_entries.append(keep_e)
                    target_other = keep_e[3] & 0x7FFF
                else:
                    target_other = 1  # VER_NDX_GLOBAL

                for he in high_entries:
                    remap_ver_idx[he[3] & 0x7FFF] = target_other

                # Rebuild the linked list of kept Vernaux entries in-place
                if low_entries:
                    first_aux_rel = low_entries[0][0] - vn_pos
                    struct.pack_into("<HHIII", data, vn_pos, vn_version, len(low_entries), vn_file, first_aux_rel, vn_next)
                    for idx_l, le in enumerate(low_entries):
                        nxt_rel = (low_entries[idx_l + 1][0] - le[0]) if idx_l + 1 < len(low_entries) else 0
                        struct.pack_into("<IHHII", data, le[0], le[1], le[2], le[3], le[4], nxt_rel)
                    # Zero out unlinked high Vernaux slots cleanly
                    for he in high_entries:
                        data[he[0] : he[0] + 16] = b"\x00" * 16

        if vn_next == 0:
            break
        vn_pos += vn_next

    # Rename any `__isoc23_<fn>` undefined symbols in `.dynsym` to `<fn>` (`st_name += 9`, exact suffix in `.dynstr`!)
    ds_off, ds_size, ds_entsize = dynsym_sh[2][4], dynsym_sh[2][5], (dynsym_sh[2][9] or 24)
    vs_off, vs_size = versym_sh[2][4], versym_sh[2][5]
    n_syms = min(ds_size // ds_entsize, vs_size // 2)

    for sym_i in range(n_syms):
        s_pos = ds_off + sym_i * ds_entsize
        st_name, st_info, st_other, st_shndx, st_val, st_sz = struct.unpack_from("<IBBHQQ", data, s_pos)
        if st_shndx == 0 and st_name < dynstr_size:  # SHN_UNDEF
            sname = _read_cstr(dynstr, st_name)
            if sname.startswith(ISOC23_PREFIX) and len(sname) > len(ISOC23_PREFIX):
                struct.pack_into("<I", data, s_pos, st_name + len(ISOC23_PREFIX))
                changed = True

        if remap_ver_idx:
            v_pos = vs_off + sym_i * 2
            vs_val = struct.unpack_from("<H", data, v_pos)[0]
            ver_id = vs_val & 0x7FFF
            if ver_id in remap_ver_idx:
                new_vs = (vs_val & 0x8000) | remap_ver_idx[ver_id]
                struct.pack_into("<H", data, v_pos, new_vs)
                changed = True

    if not changed:
        return False

    st_mode = p.stat().st_mode
    tmp = p.with_name(f".{p.name}.__glibc_fix_{os.getpid()}")
    try:
        with tmp.open("wb") as f_out:
            f_out.write(data)
        os.chmod(tmp, st_mode)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return True


def heal_runtime_elf_tree(third_party: str | Path, max_glibc: tuple[int, ...] = (2, 34)) -> int:
    """Scan `third_party/{alembic,fbx}/.venv` and downgrade any `GLIBC > 2.34` ELF64 binaries in-place."""
    tp = Path(third_party)
    if not tp.is_dir():
        return 0
    healed = 0
    candidates: list[Path] = []
    for sub in ("alembic/.venv", "fbx/.venv"):
        base = tp / sub
        if not base.is_dir():
            continue
        bin_d = base / "bin"
        if bin_d.is_dir():
            candidates.extend(p for p in bin_d.iterdir() if not p.is_symlink() and p.is_file())
        lib_d = base / "lib"
        if lib_d.is_dir():
            candidates.extend(p for p in lib_d.rglob("*.so*") if not p.is_symlink() and p.is_file())

    for cand in candidates:
        if heal_elf_glibc_compat(cand, max_glibc=max_glibc):
            healed += 1
    return healed
