"""Multi-Format VFX Image Sequence Ingestion & OCIO sRGB Conversion for ShotSpace.

Strictly image-sequence oriented (matching real undistorted matchmove plates from Nuke / 3DEqualizer / PFTrack):
  - Supported formats via `OpenImageIO`:
      * `.exr` (16-bit half / 32-bit float OpenEXR, multi-channel RGB/RGBA, handles overscan `dataWindow` vs
        `displayWindow` accurately! Defaults to `ACEScg` -> OCIO `sRGB`)
      * `.dpx`, `.cin` (10/12/16-bit film/log scans, defaults to `ADX10` -> OCIO `sRGB`)
      * `.tif`, `.tiff` (8/16-bit sRGB or 32-bit float `ACEScg`)
      * `.png` (8/16-bit), `.jpg`, `.jpeg`, `.tga`, `.bmp`, `.webp`, `.hdr`
  - Also supports extracting `.zip` / `.tar` sequence archives into the workspace sequence folder.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile

import numpy as np

from shotspace.core.platform_compat import replace_file
import OpenImageIO as oiio

from ..core.units import DEFAULT_FPS
from .ocio_color import OcioSettings, convert_frame_to_srgb_u8, resolve_input_colorspace
from .plate_grade import GradeParams

SEQUENCE_EXTS = {
    ".exr",
    ".dpx",
    ".cin",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".bmp",
    ".webp",
    ".hdr",
}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v", ".wmv"}
_FRAME_RE = re.compile(r"(?:[._-]|^)(\d{1,7})(?=\.[^.]+$)")
_SEP_FRAME_RE = re.compile(r"^(.*?[._-])(\d{1,7})(\.[^.]+)$")
_PLAIN_FRAME_RE = re.compile(r"^(\D*)(\d{1,7})(\.[^.]+)$")


def format_frame_ranges(frames: list[int]) -> str:
    """Format sorted integer frame numbers into compact ranges, e.g. `[1005, 1006, 1007, 1012]` -> `'1005–1007, 1012'`."""
    if not frames:
        return ""
    vals = sorted(set(int(x) for x in frames))
    parts: list[str] = []
    s = vals[0]
    e = vals[0]
    for v in vals[1:]:
        if v == e + 1:
            e = v
        else:
            parts.append(f"{s}" if s == e else f"{s}–{e}")
            s = e = v
    parts.append(f"{s}" if s == e else f"{s}–{e}")
    if len(parts) > 8:
        return ", ".join(parts[:8]) + f" 等共 {len(vals)} 帧"
    return ", ".join(parts)


def split_sequence_frame_token(filename: str) -> tuple[str, int] | None:
    """Extract `(normalized_stem, frame_number)` from a sequence filename, or `None` if no frame token exists."""
    m = _SEP_FRAME_RE.match(filename)
    if not m:
        m = _PLAIN_FRAME_RE.match(filename)
    if not m:
        return None
    raw_prefix = m.group(1)
    norm_stem = re.sub(r"[._-]+$", "", raw_prefix).lower()
    return norm_stem, int(m.group(2))


@dataclass
class PlateInfo:
    source_name: str
    width: int
    height: int
    fps: float
    frames: list[int]
    raw_files: dict[int, str]  # frame -> original uploaded sequence file (.exr / .dpx / .png / ...)
    preview_files: dict[int, str]  # frame -> OCIO-converted 8-bit sRGB file for WebGL backplate
    file_ext: str  # e.g. ".exr", ".png", ".dpx"
    bit_depth: str  # e.g. "half (16-bit float)", "float (32-bit)", "uint8", "uint16"
    input_colorspace: str  # user setting ("auto" or explicit)
    resolved_colorspace: str  # actual OCIO colorspace used (e.g. "ACEScg")
    ocio_view: str  # "aces_sdr" | "un_tone_mapped"
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "kind": "sequence",
            "width": self.width,
            "height": self.height,
            "fps": round(float(self.fps), 4),
            "frames": self.frames,
            "first_frame": self.frames[0] if self.frames else 1,
            "last_frame": self.frames[-1] if self.frames else 1,
            "frame_count": len(self.frames),
            "file_ext": self.file_ext,
            "bit_depth": self.bit_depth,
            "input_colorspace": self.input_colorspace,
            "resolved_colorspace": self.resolved_colorspace,
            "ocio_view": self.ocio_view,
            "warnings": list(self.warnings),
        }


def read_image_rgb(path: str | Path) -> tuple[np.ndarray, str, bool]:
    """Read any VFX image format via `OpenImageIO` into `(rgb_array, bit_depth_label, is_float)`.
    - Properly crops/pads `.exr` `dataWindow` to `displayWindow` (`full_width x full_height`) so Nuke bounding-box
      / overscan `.exr` plates align 1:1 with the camera gate!
    - Picks `R, G, B` channels by name when present (handling multi-layer EXRs whose channel 0 might be Z or A).
    """
    p = Path(path)
    buf = oiio.ImageBuf(str(p))
    if buf.has_error:
        raise ValueError(f"无法读取序列帧 {p.name}：{buf.geterror()}")

    spec = buf.spec()
    fmt = str(spec.format)
    is_float = any(k in fmt.lower() for k in ("half", "float", "double"))
    if "half" in fmt.lower():
        bd_label = "16-bit half float"
    elif "float" in fmt.lower():
        bd_label = "32-bit float"
    elif "uint16" in fmt.lower():
        bd_label = "16-bit uint"
    elif "uint10" in fmt.lower() or "uint12" in fmt.lower():
        bd_label = fmt
    else:
        bd_label = "8-bit uint"

    # Select R, G, B channels cleanly (handles RGBA or multi-AOV EXR)
    ch_names = list(spec.channelnames)
    rgb_idx: tuple[int, ...] | None = None
    upper_map = {n.upper(): i for i, n in enumerate(ch_names)}
    if all(c in upper_map for c in ("R", "G", "B")):
        rgb_idx = (upper_map["R"], upper_map["G"], upper_map["B"])
    elif spec.nchannels >= 3:
        rgb_idx = (0, 1, 2)
    elif spec.nchannels >= 1:
        rgb_idx = (0, 0, 0)
    if rgb_idx and (spec.nchannels != 3 or rgb_idx != (0, 1, 2)):
        buf = oiio.ImageBufAlgo.channels(buf, rgb_idx)
        spec = buf.spec()

    # Conform EXR dataWindow (ROI) to displayWindow (full ROI) if Nuke wrote a cropped bbox or overscan dod!
    roi_data = buf.roi
    roi_full = buf.roi_full
    if (
        roi_full.defined
        and roi_full.width > 0
        and roi_full.height > 0
        and (
            roi_data.xbegin != roi_full.xbegin
            or roi_data.ybegin != roi_full.ybegin
            or roi_data.width != roi_full.width
            or roi_data.height != roi_full.height
        )
    ):
        conformed = oiio.ImageBuf(
            oiio.ImageSpec(roi_full.width, roi_full.height, 3, oiio.FLOAT if is_float else spec.format)
        )
        oiio.ImageBufAlgo.zero(conformed)
        # Shift dataWindow origin to (0, 0) relative to displayWindow
        pixels_roi = buf.get_pixels(oiio.FLOAT if is_float else (oiio.UINT16 if "16" in fmt else oiio.UINT8), roi_full)
        arr = np.asarray(pixels_roi)
    else:
        target_type = oiio.FLOAT if (is_float or p.suffix.lower() in (".dpx", ".cin")) else (
            oiio.UINT16 if "16" in fmt else oiio.UINT8
        )
        arr = np.asarray(buf.get_pixels(target_type))

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return np.ascontiguousarray(arr[:, :, :3]), bd_label, (is_float or p.suffix.lower() in (".dpx", ".cin"))


def write_jpeg_rgb(path: str | Path, rgb_u8: np.ndarray, quality: int = 94) -> Path:
    import uuid as _uuid

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}_{_uuid.uuid4().hex[:8]}{out.suffix}")
    arr = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    spec = oiio.ImageSpec(w, h, 3, oiio.UINT8)
    spec.attribute("CompressionQuality", int(quality))
    writer = oiio.ImageOutput.create(str(tmp))
    if writer is None or not writer.open(str(tmp), spec):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"无法写入预览图：{out}")
    writer.write_image(arr)
    writer.close()
    replace_file(tmp, out)
    return out


def write_png_rgb(path: str | Path, rgb_u8: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    spec = oiio.ImageSpec(w, h, 3, oiio.UINT8)
    spec.attribute("png:compressionLevel", 3)
    writer = oiio.ImageOutput.create(str(out))
    if writer is None or not writer.open(str(out), spec):
        raise RuntimeError(f"无法写入 PNG 帧：{out}")
    writer.write_image(arr)
    writer.close()
    return out


def write_exr_half_rgb(path: str | Path, rgb_f32: np.ndarray, colorspace: str = "ACEScg") -> Path:
    """Write a 16-bit half-float ZIP-compressed `.exr` frame with standard `oiio:ColorSpace` metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb_f32, dtype=np.float32)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    spec = oiio.ImageSpec(w, h, 3, oiio.HALF)
    spec.attribute("compression", "zips")
    spec.attribute("oiio:ColorSpace", colorspace)
    writer = oiio.ImageOutput.create(str(out))
    if writer is None or not writer.open(str(out), spec):
        raise RuntimeError(f"无法写入 EXR 文件：{out}")
    writer.write_image(arr)
    writer.close()
    return out


def _decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    """Decode zip entry names safely across UTF-8 and Chinese Windows Explorer GBK/GB18030 archives,
    normalizing Windows backslash `\\` path separators to `/`."""
    raw_name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            raw_bytes = raw_name.encode("cp437")
            try:
                raw_name = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_name = raw_bytes.decode("gb18030")
        except Exception:
            pass
    return raw_name.replace("\\", "/")


def _archive_member_leaf(norm_path: str) -> str | None:
    parts = [p for p in norm_path.replace("\\", "/").split("/") if p and p != "."]
    if not parts:
        return None
    for part in parts[:-1]:
        if part.upper() == "__MACOSX" or part.startswith("."):
            return None
    leaf = parts[-1]
    if leaf.startswith(".") or leaf.lower() in ("thumbs.db", "desktop.ini"):
        return None
    if Path(leaf).suffix.lower() not in SEQUENCE_EXTS:
        return None
    return leaf


def extract_sequence_archive(archive_path: Path, dest_dir: Path) -> int:
    """Safely extract supported image sequence files from a `.zip` or `.tar` archive into `dest_dir`
    (fully compatible with Windows backslash separators, GBK Chinese zip archives, and macOS/Windows junk files)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                leaf = _archive_member_leaf(_decode_zip_member_name(info))
                if not leaf:
                    continue
                target = dest_dir / leaf
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    elif suffix in (".tar", ".tgz") or archive_path.name.lower().endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                leaf = _archive_member_leaf(member.name)
                if not leaf:
                    continue
                extracted = tf.extractfile(member)
                if extracted is not None:
                    with extracted, (dest_dir / leaf).open("wb") as dst:
                        shutil.copyfileobj(extracted, dst)
                    count += 1
    return count


def _parse_frame_number(filename: str, fallback: int) -> int:
    tok = split_sequence_frame_token(filename)
    return tok[1] if tok is not None else fallback


def _detect_sequence_prefix(files: list[Path]) -> str:
    if not files:
        return "sequence"
    first = files[0].name
    m = _SEP_FRAME_RE.match(first) or _PLAIN_FRAME_RE.match(first)
    if m and m.group(1):
        stem = m.group(1).rstrip("._-")
        if stem:
            return f"{stem}.####{files[0].suffix.lower()}"
    return f"####{files[0].suffix.lower()}"


def _inspect_frame_spec(path: Path) -> tuple[int, int, str, bool, str | None]:
    """Fast OpenImageIO header + non-empty check returning `(eff_w, eff_h, bd_label, is_float, error_msg)`."""
    try:
        sz = path.stat().st_size
    except OSError as exc:
        return 0, 0, "", False, f"{path.name}（无法读取文件：{exc}）"
    if sz <= 0:
        return 0, 0, "", False, f"{path.name}（空文件 0 字节）"

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        err = oiio.geterror() or "格式损坏或无法识别"
        return 0, 0, "", False, f"{path.name}（{err.strip()}）"
    try:
        spec = inp.spec()
        eff_w = int(spec.full_width if spec.full_width > 0 else spec.width)
        eff_h = int(spec.full_height if spec.full_height > 0 else spec.height)
        nch = int(spec.nchannels)
        fmt = str(spec.format)
    finally:
        inp.close()

    if eff_w < 16 or eff_h < 16 or nch < 1:
        return eff_w, eff_h, "", False, f"{path.name}（异常尺寸 {eff_w}×{eff_h} / {nch} 通道）"

    is_float = any(k in fmt.lower() for k in ("half", "float", "double"))
    if "half" in fmt.lower():
        bd_label = "16-bit half float"
    elif "float" in fmt.lower():
        bd_label = "32-bit float"
    elif "uint16" in fmt.lower():
        bd_label = "16-bit uint"
    elif "uint10" in fmt.lower() or "uint12" in fmt.lower():
        bd_label = fmt
    else:
        bd_label = "8-bit uint"
    return eff_w, eff_h, bd_label, is_float, None


def ingest_sequence_dir(
    seq_dir: str | Path,
    cache_dir: str | Path,
    ocio: OcioSettings | None = None,
    fps: float = DEFAULT_FPS,
    start_frame: int = 1001,
) -> PlateInfo:
    """Validate & ingest a folder of sequence frames with strict VFX quality gates:
      - Intercepts unsupported/video-only folders, `< 2` single-image mistakes, 0-byte/corrupted frames,
        unnumbered files, mixed sequence prefixes, duplicate frame numbers, missing frames (`缺帧`),
        and per-frame resolution mismatches (`某帧分辨率不一致`) BEFORE writing `preview_srgb`.
      - Applies OCIO -> `sRGB` conversion in parallel across CPU threads and returns `PlateInfo`."""
    sdir = Path(seq_dir).resolve()
    cdir = Path(cache_dir).resolve()

    all_non_hidden = [
        p
        for p in sdir.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.name.lower() not in ("thumbs.db", "desktop.ini")
        and "__MACOSX" not in p.parts
    ]
    candidates = [p for p in all_non_hidden if p.suffix.lower() in SEQUENCE_EXTS]
    if not candidates:
        videos = [p.name for p in all_non_hidden if p.suffix.lower() in VIDEO_EXTS]
        if videos:
            raise ValueError(
                f"素材检测拦截：暂不支持直接导入视频文件（检测到 {videos[0]}），请上传去畸变序列图文件夹或 .zip/.tar 序列包"
            )
        raise ValueError(
            f"素材检测拦截：未检测到有效的序列帧图片（支持格式：{', '.join(sorted(SEQUENCE_EXTS))}）"
        )

    warnings: list[str] = []
    ext_counts: dict[str, int] = {}
    for p in candidates:
        ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
    dom_ext = max(ext_counts, key=ext_counts.get)
    other_exts = {k: v for k, v in ext_counts.items() if k != dom_ext}

    dom_files = [p for p in candidates if p.suffix.lower() == dom_ext]
    if len(dom_files) < 2:
        only_name = dom_files[0].name if dom_files else ""
        extra_hint = (
            f"（另检测到其他格式 {', '.join(f'{k}×{v}' for k, v in other_exts.items())}，序列帧格式必须统一）"
            if other_exts
            else ""
        )
        raise ValueError(
            f"素材检测拦截：序列解算至少需要连续 2 帧以上，当前仅检测到 {len(dom_files)} 帧图片"
            f"{f'（{only_name}）' if only_name else ''}{extra_hint}；请拖入完整序列文件夹或全选所有序列帧。"
        )

    # 1. Validate frame tokens, single sequence prefix, duplicate frames, and missing frames (`缺帧`)
    unnumbered: list[str] = []
    stems_map: dict[str, list[Path]] = {}
    parsed_items: list[tuple[str, int, Path]] = []
    for p in dom_files:
        tok = split_sequence_frame_token(p.name)
        if tok is None:
            unnumbered.append(p.name)
        else:
            stem, fnum = tok
            stems_map.setdefault(stem, []).append(p)
            parsed_items.append((stem, fnum, p))

    if unnumbered:
        sample = ", ".join(sorted(unnumbered)[:4])
        more = f" 等 {len(unnumbered)} 个文件" if len(unnumbered) > 4 else ""
        raise ValueError(
            f"素材检测拦截：检测到无帧号命名的图片混入序列（{sample}{more}），请清理非序列单张图后重新上传。"
        )

    if len(stems_map) > 1:
        stem_desc = "、".join(
            f"{(st or '<纯数字>')}* ({len(ps)}帧)"
            for st, ps in sorted(stems_map.items(), key=lambda kv: -len(kv[1]))[:4]
        )
        raise ValueError(
            f"素材检测拦截：同一文件夹内混入了多个不同前缀的序列（{stem_desc}），请确保每次仅上传单个镜头的序列图。"
        )

    frame_to_paths: dict[int, list[Path]] = {}
    for _, fnum, p in parsed_items:
        frame_to_paths.setdefault(fnum, []).append(p)

    dup_frames = sorted(f for f, ps in frame_to_paths.items() if len(ps) > 1)
    if dup_frames:
        f0 = dup_frames[0]
        dup_names = ", ".join(x.name for x in frame_to_paths[f0][:3])
        raise ValueError(
            f"素材检测拦截：检测到重复帧号（共 {len(dup_frames)} 处：{format_frame_ranges(dup_frames)}，"
            f"例如帧 {f0} 存在多个文件：{dup_names}），请清理重复文件后重新上传。"
        )

    used_frames = sorted(frame_to_paths.keys())
    min_f, max_f = used_frames[0], used_frames[-1]
    expected_total = max_f - min_f + 1
    if expected_total != len(used_frames):
        missing = sorted(set(range(min_f, max_f + 1)) - set(used_frames))
        other_note = (
            f"（同时发现其他后缀文件 {', '.join(f'{k}×{v}' for k, v in other_exts.items())}，请检查是否有帧渲染成了不同格式）"
            if other_exts
            else ""
        )
        raise ValueError(
            f"素材检测拦截：序列存在缺帧！帧范围 {min_f}–{max_f} 应有 {expected_total} 帧，实传 {len(used_frames)} 帧，"
            f"缺失 {len(missing)} 帧：{format_frame_ranges(missing)}{other_note}。"
        )

    if other_exts:
        warnings.append(
            f"文件夹内混有其他格式图片（{', '.join(f'{k} {v}个' for k, v in sorted(other_exts.items()))}），已自动忽略并加载主序列 {dom_ext}（{len(used_frames)} 帧）"
        )

    files = [frame_to_paths[f][0] for f in used_frames]
    raw_files: dict[int, str] = {f: str(frame_to_paths[f][0]) for f in used_frames}

    # 2. Parallel per-frame OpenImageIO header + resolution consistency + corruption check across ALL frames!
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(used_frames)))) as pool:
        specs_list = list(pool.map(lambda f: (f, *_inspect_frame_spec(frame_to_paths[f][0])), used_frames))

    corrupted = [err for _, _, _, _, _, err in specs_list if err is not None]
    if corrupted:
        sample_err = "；".join(corrupted[:3])
        more_err = f"（共 {len(corrupted)} 帧损坏）" if len(corrupted) > 3 else ""
        raise ValueError(f"素材检测拦截：发现损坏或无法读取的序列帧：{sample_err}{more_err}。")

    _, w, h, bd_label, is_float, _ = specs_list[0]
    res_mismatches = [
        (f, frame_to_paths[f][0].name, fw, fh)
        for f, fw, fh, _, _, _ in specs_list[1:]
        if (fw, fh) != (w, h)
    ]
    if res_mismatches:
        sample_mm = "、".join(f"帧 {f} ({nm}: {fw}×{fh})" for f, nm, fw, fh in res_mismatches[:4])
        more_mm = f" 等共 {len(res_mismatches)} 帧" if len(res_mismatches) > 4 else ""
        raise ValueError(
            f"素材检测拦截：序列帧分辨率不一致！首帧 {used_frames[0]} ({files[0].name}) 为 {w}×{h}，"
            f"发现尺寸不一致帧：{sample_mm}{more_mm}。"
        )

    bd_mismatches = [f for f, _, _, bd, _, _ in specs_list[1:] if bd != bd_label]
    if bd_mismatches:
        warnings.append(
            f"部分帧位深与首帧 ({bd_label}) 不同（帧 {format_frame_ranges(bd_mismatches)}），已自动统一转换至 sRGB"
        )

    first_arr, bd_label, is_float = read_image_rgb(files[0])
    h, w = int(first_arr.shape[0]), int(first_arr.shape[1])

    ocio_cfg = ocio or OcioSettings()
    resolved_cs = resolve_input_colorspace(ocio_cfg.input_colorspace, dom_ext, is_float)

    # Parallel OCIO -> sRGB preview baking (verifies full pixel decode across ALL frames even on passthrough!)
    can_passthrough = (
        resolved_cs in ("sRGB Encoded Rec.709 (sRGB)", "Raw")
        and dom_ext in (".jpg", ".jpeg", ".png", ".webp")
        and not is_float
        and first_arr.dtype == np.uint8
        and not bd_mismatches
    )

    tmp_preview_dir = cdir / ".preview_srgb_staging"
    if tmp_preview_dir.exists():
        shutil.rmtree(tmp_preview_dir, ignore_errors=True)

    preview_files: dict[int, str] = {}
    try:
        if can_passthrough:
            def _verify_pixels(fnum: int) -> None:
                if fnum == used_frames[0]:
                    return
                p_cur = Path(raw_files[fnum])
                buf = oiio.ImageBuf(str(p_cur))
                if not buf.read(0, 0, True, oiio.UINT8) or buf.has_error:
                    raise ValueError(f"素材检测拦截：序列帧像素数据不完整或损坏：{p_cur.name}（{buf.geterror()}）")

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(used_frames)))) as pool:
                list(pool.map(_verify_pixels, used_frames))
            preview_files = dict(raw_files)
            preview_dir = cdir / "preview_srgb"
            if preview_dir.exists():
                shutil.rmtree(preview_dir, ignore_errors=True)
            preview_dir.mkdir(parents=True, exist_ok=True)
        else:
            tmp_preview_dir.mkdir(parents=True, exist_ok=True)

            def _convert_one(fnum: int) -> tuple[int, str]:
                arr = first_arr if fnum == used_frames[0] else read_image_rgb(raw_files[fnum])[0]
                if int(arr.shape[1]) != w or int(arr.shape[0]) != h:
                    raise ValueError(
                        f"素材检测拦截：序列帧 {fnum} ({Path(raw_files[fnum]).name}) 解码尺寸 {arr.shape[1]}×{arr.shape[0]} 与首帧 {w}×{h} 不一致！"
                    )
                u8 = convert_frame_to_srgb_u8(arr, resolved_cs, ocio_cfg.ocio_view)
                out_p = write_jpeg_rgb(tmp_preview_dir / f"srgb_{fnum:06d}.jpg", u8, quality=95)
                return fnum, out_p.name

            baked_names: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(used_frames)))) as pool:
                for fnum, out_name in pool.map(_convert_one, used_frames):
                    baked_names[fnum] = out_name

            preview_dir = cdir / "preview_srgb"
            if preview_dir.exists():
                shutil.rmtree(preview_dir, ignore_errors=True)
            tmp_preview_dir.replace(preview_dir)
            preview_files = {fnum: str(preview_dir / baked_names[fnum]) for fnum in used_frames}
    except Exception:
        if tmp_preview_dir.exists():
            shutil.rmtree(tmp_preview_dir, ignore_errors=True)
        raise

    return PlateInfo(
        source_name=_detect_sequence_prefix(files),
        width=w,
        height=h,
        fps=fps,
        frames=used_frames,
        raw_files=raw_files,
        preview_files=preview_files,
        file_ext=dom_ext,
        bit_depth=bd_label,
        input_colorspace=ocio_cfg.input_colorspace,
        resolved_colorspace=resolved_cs,
        ocio_view=ocio_cfg.ocio_view,
        warnings=warnings,
    )


def prepare_solver_srgb_frames(
    plate: PlateInfo,
    grade: GradeParams,
    out_dir: Path,
) -> dict[int, str]:
    """Prepare exact 8-bit `sRGB` PNG frames for `ViTDet-H`, `SAM 3.1`, `SAM 3D Body` & `SAM 3D Objects`:
    - When `plate` is `ACEScg` `.exr` / `ADX10` `.dpx` / 16-bit, applies `grade.exposure` in scene-linear `ACEScg`
      FIRST, then runs OCIO -> `sRGB`, then `brightness`, `contrast`, `gamma`, and `saturation`, writing lossless
      8-bit PNGs for all GPU workers!
    - When `plate` is already 8-bit `sRGB` and `grade.is_identity`, passes `preview_files` directly with zero copy!
    """
    g_clamped = grade.clamped()
    if (
        plate.resolved_colorspace in ("sRGB Encoded Rec.709 (sRGB)", "Raw")
        and "8-bit" in plate.bit_depth
        and g_clamped.is_identity
    ):
        return dict(plate.preview_files)

    out_dir.mkdir(parents=True, exist_ok=True)
    sig_file = out_dir / "_solver_grade_sig.json"
    want_sig = json.dumps(
        {
            "engine": "ocio_v2",
            "cs": plate.resolved_colorspace,
            "view": plate.ocio_view,
            "grade": g_clamped.to_dict(),
            "frames": [int(f) for f in plate.frames],
        },
        sort_keys=True,
    )
    if sig_file.is_file():
        try:
            if sig_file.read_text(encoding="utf-8") == want_sig:
                cached = {int(f): str(out_dir / f"solve_{int(f):06d}.png") for f in plate.frames}
                if all(Path(p).is_file() for p in cached.values()):
                    return cached
        except OSError:
            pass

    result: dict[int, str] = {}

    def _bake(fnum: int) -> tuple[int, str]:
        arr, _, _ = read_image_rgb(plate.raw_files[fnum])
        u8 = convert_frame_to_srgb_u8(arr, plate.resolved_colorspace, plate.ocio_view, grade=g_clamped)
        out_p = write_png_rgb(out_dir / f"solve_{fnum:06d}.png", u8)
        return fnum, str(out_p)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(plate.frames)))) as pool:
        for fnum, p_str in pool.map(_bake, plate.frames):
            result[fnum] = p_str

    tmp_sig = out_dir / "._solver_grade_sig.tmp"
    tmp_sig.write_text(want_sig, encoding="utf-8")
    tmp_sig.replace(sig_file)
    return result
