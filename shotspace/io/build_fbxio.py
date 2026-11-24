"""Rebuild `fbxio` C++ pybind11 extension against Autodesk FBX SDK 2020.3.10 on Rocky Linux 9.8 (gcc 11.5)."""

from __future__ import annotations

import shutil
import subprocess
import sysconfig
from pathlib import Path

from shotspace.config import get_paths


def build_fbxio() -> Path:
    paths = get_paths()
    py = paths.fbx_python
    sdk = paths.fbx_sdk
    cpp = Path(__file__).resolve().parent / "workers" / "fbxio.cpp"
    if not cpp.is_file():
        raise FileNotFoundError(f"Missing {cpp}")

    # Query target venv's pybind11 includes & extension suffix
    info_cmd = [
        str(py),
        "-c",
        "import json, pybind11, sysconfig; print(json.dumps({"
        "'inc': pybind11.get_include(), "
        "'pyinc': sysconfig.get_path('include'), "
        "'purelib': sysconfig.get_path('purelib'), "
        "'ext': sysconfig.get_config_var('EXT_SUFFIX')"
        "}))",
    ]
    import json

    meta = json.loads(subprocess.check_output(info_cmd, text=True))
    out_so = Path(meta["purelib"]) / f"fbxio{meta['ext']}"

    # Static libfbxsdk.a + system libxml2 & zlib
    lib_dir = sdk / "lib" / "gcc" / "x64" / "release"
    if not lib_dir.is_dir():
        lib_dir = next(sdk.rglob("libfbxsdk.a")).parent

    gxx = shutil.which("g++") or "g++"
    cmd = [
        gxx,
        "-O3",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-I{meta['pyinc']}",
        f"-I{meta['inc']}",
        f"-I{sdk / 'include'}",
        str(cpp),
        str(lib_dir / "libfbxsdk.a"),
        "-lxml2",
        "-lz",
        "-ldl",
        "-lpthread",
        "-o",
        str(out_so),
    ]
    subprocess.run(cmd, check=True)
    return out_so


if __name__ == "__main__":
    built = build_fbxio()
    print(f"Built {built}")
