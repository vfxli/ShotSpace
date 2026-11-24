"""ShotSpace installer wizard (started by setup.sh / setup.bat; standard library only).

Puts everything ShotSpace needs inside the ShotSpace folder:

    .venv                              the service (Python 3.12)
    third_party/sam_3d_body/           SAM 3D Body source, its worker environment (Python 3.11), its weights
    third_party/sam_3d_objects/        SAM 3.1 source (+ ShotSpace's patch), the tracking worker environment,
                                       its weights; with object matching also SAM 3D Objects, MoGe, TripoSR
    third_party/alembic, fbx/          the Alembic / FBX export environments (prebuilt native modules)
    .tools/                            uv, Python, caches, downloads in progress, the installer's own record

Every download is checked against the checksum recorded in install/sources.json and, when one source fails,
the next one is tried.  Interrupted downloads continue where they stopped.  Running the installer again only does
what is missing.

    setup.sh  [--lang zh|en] [--yes] [--region auto|cn|global] [--weights modelscope|huggingface|hf-mirror|DIR]
              [--objects] [--no-export] [--native-dir DIR] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from zlib import error as zlib_error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install"
TOOLS = ROOT / ".tools"
TP = ROOT / "third_party"
DOWNLOADS = TOOLS / "downloads"
STATE_FILE = TOOLS / "install_state.json"
IS_WIN = os.name == "nt"
SOURCES = json.loads((INSTALL / "sources.json").read_text(encoding="utf-8")) if (INSTALL / "sources.json").is_file() else {}

LANG = "zh"


def T(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


# ------------------------------------------------------------------------------------------------ printing / input
class Abort(Exception):
    pass


class TooSlow(IOError):
    pass


def say(text: str = "") -> None:
    print(text, flush=True)


def head(text: str) -> None:
    say()
    say("=" * 72)
    say(text)
    say("=" * 72)


def ask(question: str, default: str, choices: list[str] | None = None, assume_default: bool = False) -> str:
    if assume_default:
        say(f"{question} [{default}]")
        return default
    hint = "/".join(choices) if choices else ""
    while True:
        try:
            raw = input(f"{question} [{default}]{(' (' + hint + ')') if hint else ''}: ").strip()
        except EOFError:
            raw = ""
        value = raw or default
        if not choices or value in choices:
            return value
        say(T("请输入以下之一：", "Please answer one of: ") + hint)


def ask_yes(question: str, default: bool, assume_default: bool = False) -> bool:
    d = "y" if default else "n"
    return ask(question, d, ["y", "n"], assume_default) == "y"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


# ------------------------------------------------------------------------------------------------ state
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ------------------------------------------------------------------------------------------------ hashing
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """Checksum of a source tree independent of how it was packed: every file's path and content, in order."""
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                continue
            digest = file_sha256(p)
            items.append(f"{rel}\0{digest}\n")
    h = hashlib.sha256()
    for line in sorted(items):
        h.update(line.encode("utf-8"))
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ network
_CTX = ssl.create_default_context()
UA = "ShotSpace-installer"


def _open(url: str, headers: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout, context=_CTX)


def probe(url: str, timeout: float = 8.0) -> float | None:
    """Seconds to the first bytes of `url`, None when it does not answer."""
    t0 = time.time()
    try:
        with _open(url, {"Range": "bytes=0-1023"}, timeout=timeout) as resp:
            resp.read(1024)
        return time.time() - t0
    except Exception:  # noqa: BLE001 - any failure means: not usable now
        return None


def download(url: str, dest: Path, sha256: str | None, size: int | None = None, headers: dict | None = None,
             label: str = "", min_rate: bool = True) -> None:
    """Fetch `url` to `dest` (resuming a partial download), check its checksum.  Raises on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    hdrs = dict(headers or {})
    if have:
        hdrs["Range"] = f"bytes={have}-"
    if not (size and have == size):
        try:
            resp = _open(url, hdrs, timeout=60.0)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:
                resp = None
            else:
                raise
        if resp is not None:
            with resp:
                if have and resp.status != 206:     # the server ignored the range: start over
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = size or (have + int(resp.headers.get("Content-Length") or 0)) or None
                done, t0, last = have, time.time(), 0.0
                win_t, win_b = t0, done
                with part.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if min_rate and now - win_t > 30.0:      # slower than 64 KB/s for half a minute: try the next source
                            if done - win_b < 30 * 64 * 1024:
                                sys.stdout.write("\n")
                                raise TooSlow(T("太慢，换下一个源（已下载的部分保留）", "too slow, trying the next source (kept what arrived)"))
                            win_t, win_b = now, done
                        if now - last > 0.5:
                            last = now
                            rate = (done - have) / max(now - t0, 1e-6)
                            pct = f"{100.0 * done / total:5.1f}%" if total else ""
                            sys.stdout.write(f"\r    {label} {pct} {human(done)}" + (f" / {human(total)}" if total else "")
                                             + f"  {human(rate)}/s   ")
                            sys.stdout.flush()
                sys.stdout.write("\n")
    got = part.stat().st_size
    if size and got != size:
        raise IOError(T(f"大小不对：{got}，应为 {size}", f"size {got}, expected {size}"))
    if sha256:
        digest = file_sha256(part)
        if digest != sha256:
            part.unlink()
            raise IOError(T("校验不一致（文件已删除，会从下一个源重新下载）", "checksum mismatch (file removed)"))
    part.replace(dest)


def fetch_any(urls: list[str], dest: Path, sha256: str | None, size: int | None = None, headers_for=None,
              label: str = "") -> None:
    """Try each source in turn until one gives the right file.  A source that crawls is left for the next one;
    when every usable source crawled, they are tried once more without the speed limit."""
    if dest.exists() and (not sha256 or file_sha256(dest) == sha256):
        return
    errors = []
    slow: list[str] = []
    for url, min_rate in [(u, True) for u in urls] + [(None, False)]:
        if url is None:
            if not slow:
                break
            for u in slow:
                if _fetch_one(u, dest, sha256, size, headers_for, label, False, errors):
                    return
            break
        res = _fetch_one(url, dest, sha256, size, headers_for, label, True, errors)
        if res is True:
            return
        if res == "slow":
            slow.append(url)
    raise Abort(T("所有下载源都失败了：", "every source failed: ") + "; ".join(errors[-6:]))


def _fetch_one(url, dest, sha256, size, headers_for, label, min_rate, errors):
    host = urllib.parse.urlsplit(url).netloc
    say(f"    <- {host}")
    for attempt in range(3):
        try:
            download(url, dest, sha256, size, headers_for(url) if headers_for else None, label, min_rate)
            return True
        except TooSlow as exc:
            errors.append(f"{host}: {exc}")
            say(f"    {host}: {exc}")
            return "slow"
        except urllib.error.HTTPError as exc:
            errors.append(f"{host}: HTTP {exc.code}")
            say(f"    {host}: HTTP {exc.code}")
            if exc.code in (401, 403, 404):
                return False
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{host}: {exc}")
            say(f"    {host}: {exc}")
        time.sleep(2 * (attempt + 1))
    return False


# ------------------------------------------------------------------------------------------------ mirrors
def order_mirrors(group: str, region: str) -> list[dict]:
    """Sources of a group, best first.  'cn' / 'global' put that region's first; 'auto' measures."""
    mirrors = list(SOURCES["mirrors"][group])
    if region in ("cn", "global"):
        mirrors.sort(key=lambda m: 0 if m.get("region") in (region, "any") else 1)
        return mirrors
    timed = []
    for m in mirrors:
        t = probe(m["probe"])
        timed.append((t if t is not None else 1e9, m))
    timed.sort(key=lambda tm: tm[0])
    return [m for _t, m in timed]


# ------------------------------------------------------------------------------------------------ processes
def run(cmd: list[str], env: dict | None = None, cwd: Path | None = None, check: bool = True) -> int:
    say("    $ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], env=env, cwd=str(cwd) if cwd else None)
    if check and proc.returncode != 0:
        raise Abort(T(f"命令失败（{proc.returncode}）", f"command failed ({proc.returncode})"))
    return proc.returncode


class Uv:
    def __init__(self, exe: str, region: str) -> None:
        self.exe = exe
        self.region = region
        self.env = dict(os.environ)
        self.env.update({
            "UV_CACHE_DIR": str(TOOLS / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(TOOLS / "python"),
            "UV_MANAGED_PYTHON": "1",
            "UV_NO_CONFIG": "1",
            "UV_HTTP_TIMEOUT": "180",
            "UV_CONCURRENT_DOWNLOADS": "4",
        })
        for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "UV_INDEX_URL", "UV_DEFAULT_INDEX",
                  "UV_EXTRA_INDEX_URL", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
            self.env.pop(k, None)
        self._pypi: list[dict] | None = None
        self._torch: list[dict] | None = None

    def pypi(self) -> list[dict]:
        if self._pypi is None:
            self._pypi = order_mirrors("pypi", self.region)
        return self._pypi

    def torch_indexes(self) -> list[dict]:
        if self._torch is None:
            self._torch = order_mirrors("pytorch", self.region)
        return self._torch

    def python(self, version: str) -> None:
        if subprocess.run([self.exe, "python", "find", version], env=self.env, capture_output=True).returncode == 0:
            return
        for m in order_mirrors("python", self.region):
            args = [self.exe, "python", "install", version, "--no-bin", "--no-registry"]
            if m["url"]:
                args += ["--mirror", m["url"]]
            if run(args, self.env, check=False) == 0:
                return
        raise Abort(T(f"Python {version} 下载失败", f"could not download Python {version}"))

    def venv(self, path: Path, version: str) -> Path:
        py = venv_python(path)
        if py.exists():
            return py
        self.python(version)
        run([self.exe, "venv", str(path), "--python", version, "--seed"], self.env)
        return py

    def pip(self, py: Path, args: list[str], indexes: list[dict]) -> None:
        last = None
        for m in indexes:
            say(T(f"    使用源：{m['name']}", f"    index: {m['name']}"))
            rc = run([self.exe, "pip", "install", "--python", str(py), "--index-url", m["url"], *args], self.env,
                     check=False)
            if rc == 0:
                return
            last = rc
        raise Abort(T(f"安装 Python 包失败（{last}），所有源都试过了", f"package install failed ({last}) on every index"))


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def site_packages(py: Path) -> Path:
    out = subprocess.check_output([str(py), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True)
    return Path(out.strip())


def point_at_project(py: Path) -> None:
    """The worker environments import ShotSpace from this folder."""
    sp = site_packages(py)
    (sp / "shotspace_root.pth").write_text(f"{ROOT}\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ archives / patches
def safe_extract_tar(archive: Path, dest: Path, strip_first: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = []
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if strip_first:
                parts = parts[1:]
            if not parts:
                continue
            rel = Path(*parts)
            if rel.is_absolute() or ".." in rel.parts:
                raise Abort(f"unsafe path in archive: {m.name}")
            if not (m.isfile() or m.isdir()):   # the pinned sources hold no links; anything else is skipped
                continue
            m.name = rel.as_posix()
            members.append(m)
        kwargs = {"filter": "tar"} if sys.version_info >= (3, 12) else {}
        tf.extractall(dest, members=members, **kwargs)


def apply_patch(tree: Path, patch_file: Path) -> None:
    """Apply a unified diff exactly (no fuzz): the pinned source must match it line for line."""
    text = patch_file.read_text(encoding="utf-8").splitlines(keepends=True)
    i = 0
    while i < len(text):
        line = text[i]
        if line.startswith("--- "):
            target = text[i + 1][4:].strip()
            target = target[2:] if target.startswith("b/") else target
            i += 2
            path = tree / target
            src = path.read_text(encoding="utf-8").splitlines(keepends=True)
            out: list[str] = []
            pos = 0
            while i < len(text) and text[i].startswith("@@"):
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", text[i])
                start = int(m.group(1)) - 1
                i += 1
                out.extend(src[pos:start])
                pos = start
                while i < len(text) and not text[i].startswith(("@@", "diff ", "--- ")):
                    h = text[i]
                    if h.startswith("\\"):
                        i += 1
                        continue
                    tag, body = h[:1], h[1:]
                    if tag in (" ", "-"):
                        if pos >= len(src) or src[pos].rstrip("\r\n") != body.rstrip("\r\n"):
                            raise Abort(f"patch does not match {target} at line {pos + 1}")
                        if tag == " ":
                            out.append(src[pos])
                        pos += 1
                    elif tag == "+":
                        out.append(body)
                    i += 1
            out.extend(src[pos:])
            path.write_text("".join(out), encoding="utf-8")
        else:
            i += 1


# ------------------------------------------------------------------------------------------------ steps
class Installer:
    def __init__(self, args) -> None:
        self.args = args
        self.state = load_state()
        self.uv = Uv(args.uv, args.region)
        self.region = args.region
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()

    def done(self, key: str, fingerprint: str) -> bool:
        return self.state.get(key) == fingerprint

    def mark(self, key: str, fingerprint: str) -> None:
        self.state[key] = fingerprint
        save_state(self.state)

    # -- the service
    def main_env(self) -> None:
        req = INSTALL / "requirements" / "main.txt"
        fp = file_sha256(req) + file_sha256(ROOT / "pyproject.toml")
        if self.done("main", fp) and venv_python(ROOT / ".venv").exists():
            say(T("  已就绪", "  ready"))
            return
        py = self.uv.venv(ROOT / ".venv", "3.12")
        self.uv.pip(py, ["-r", str(req)], self.uv.pypi())
        self.uv.pip(py, ["--no-deps", "-e", str(ROOT)], self.uv.pypi())
        self.mark("main", fp)

    # -- a GPU worker environment
    def gpu_env(self, key: str, venv: Path, req_files: list[Path]) -> None:
        tv = SOURCES["torch"]
        fp = json.dumps(tv) + "".join(file_sha256(r) for r in req_files)
        if self.done(key, fp) and venv_python(venv).exists():
            say(T("  已就绪", "  ready"))
            return
        py = self.uv.venv(venv, tv["python"])
        say(T("  PyTorch（约 3 GB，第一次较慢）", "  PyTorch (about 3 GB, slow the first time)"))
        last = None
        for m in self.uv.torch_indexes():
            say(T(f"    使用源：{m['name']}", f"    index: {m['name']}"))
            rc = run([self.uv.exe, "pip", "install", "--python", str(py), "--index-url", m["url"],
                      f"torch=={tv['torch']}", f"torchvision=={tv['torchvision']}"], self.uv.env, check=False)
            if rc == 0:
                last = 0
                break
            last = rc
        if last != 0:
            raise Abort(T("PyTorch 安装失败，所有源都试过了", "PyTorch install failed on every index"))
        for r in req_files:
            self.uv.pip(py, ["-r", str(r)], self.uv.pypi())
        point_at_project(py)
        self.mark(key, fp)

    # -- pinned source trees
    def source(self, name: str) -> None:
        src = SOURCES["sources"][name]
        dest = ROOT / src["dest"]
        fp = src["commit"] + (file_sha256(INSTALL / "patches" / src["patch"]) if src.get("patch") else "")
        if self.done("src:" + name, fp) and dest.is_dir():
            say(T(f"  {name}：已就绪", f"  {name}: ready"))
            return
        say(f"  {name} @ {src['commit'][:10]}")
        archive = DOWNLOADS / f"{name}-{src['commit']}.tar.gz"
        urls = [m["url"].format(repo=src["repo"], commit=src["commit"]) for m in order_mirrors("github", self.region)]
        errors = []
        for i, url in enumerate(urls + urls):   # a second round: archives are streamed and can break off
            tmp = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=str(TOOLS / "tmp")))
            try:
                archive.unlink(missing_ok=True)
                k = i % len(urls)
                fetch_any(urls[k:] + urls[:k], archive, None, label=name)   # a slow source hands over to the next
                safe_extract_tar(archive, tmp / "tree")
                got = tree_sha256(tmp / "tree")
                if got != src["tree_sha256"]:
                    raise IOError(T(f"内容与记录不一致（{got[:12]}）", f"content differs from the record ({got[:12]})"))
                if src.get("patch"):
                    apply_patch(tmp / "tree", INSTALL / "patches" / src["patch"])
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp / "tree"), str(dest))
                break
            except (Abort, OSError, EOFError, tarfile.TarError, zlib_error) as exc:
                host = urllib.parse.urlsplit(url).netloc
                errors.append(f"{host}: {exc}")
                say(f"    {host}: {exc}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            archive.unlink(missing_ok=True)
            raise Abort(T(f"{name} 源码所有下载源都失败了：", f"{name}: every source failed: ") + "; ".join(errors))
        archive.unlink(missing_ok=True)
        self.mark("src:" + name, fp)

    # -- model weights
    def weights(self, name: str) -> None:
        w = SOURCES["weights"][name]
        dest = ROOT / w["dest"]
        choice = self.args.weights
        if choice not in ("modelscope", "huggingface", "hf-mirror"):
            local = Path(choice).expanduser()
        else:
            local = None
        for f in w["files"]:
            target = dest / f["path"]
            if target.exists() and target.stat().st_size == f["size"] and (
                    self.state.get("w:" + str(target.relative_to(ROOT))) == f["sha256"] or file_sha256(target) == f["sha256"]):
                self.state["w:" + str(target.relative_to(ROOT))] = f["sha256"]
                continue
            label = f"{name}/{f['path']}"
            if f["size"] == 0:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"")
                continue
            if local is not None:
                # accepted layouts: <folder>/<model name>/<file>, <folder>/<file>, or the third_party folder of
                # another ShotSpace installation
                tried = [local / name / f["path"], local / f["path"],
                         local / Path(w["dest"]).relative_to("third_party") / f["path"],
                         local / w["dest"] / f["path"]]
                cand = next((c for c in tried if c.is_file()), None)
                if cand is None:
                    raise Abort(T(f"本地文件夹里找不到 {name}/{f['path']}", f"not found in the folder: {name}/{f['path']}"))
                say(f"  {label} <- {cand}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cand, target)
                if file_sha256(target) != f["sha256"]:
                    target.unlink()
                    raise Abort(T(f"{cand} 校验不一致", f"{cand}: checksum mismatch"))
            else:
                urls = self.weight_urls(w, f["path"])
                say(f"  {label} ({human(f['size'])})")

                def _headers(url: str) -> dict:
                    host = urllib.parse.urlsplit(url).netloc
                    if self.hf_token and host == "huggingface.co":
                        return {"Authorization": f"Bearer {self.hf_token}"}
                    if self.hf_token and host == "hf-mirror.com" and self.args.token_to_mirror:
                        return {"Authorization": f"Bearer {self.hf_token}"}
                    return {}

                fetch_any(urls, target, f["sha256"], f["size"], _headers, label=f["path"])
            self.state["w:" + str(target.relative_to(ROOT))] = f["sha256"]
            save_state(self.state)
        save_state(self.state)

    def weight_urls(self, w: dict, path: str) -> list[str]:
        order = {"modelscope": ["modelscope", "huggingface", "hf-mirror"],
                 "huggingface": ["huggingface", "hf-mirror", "modelscope"],
                 "hf-mirror": ["hf-mirror", "huggingface", "modelscope"]}[self.args.weights]
        urls = []
        for site in order:
            repo = w.get(site)
            if not repo:
                continue
            if site in ("huggingface", "hf-mirror") and w.get("gated") and not self.hf_token:
                continue
            if site == "hf-mirror" and w.get("gated") and not self.args.token_to_mirror:
                continue
            base = SOURCES["weight_sites"][site]
            urls.append(base.format(repo=repo, rev=w["revision"].get(site, "main"), path=urllib.parse.quote(path)))
        if not urls:
            raise Abort(T("没有可用的下载源：这个模型在 HuggingFace 需要登录，请设置 HF_TOKEN 或改用 ModelScope",
                          "no usable source: this model needs a HuggingFace token (HF_TOKEN) or use ModelScope"))
        return urls

    # -- export modules
    def export_envs(self) -> None:
        nb = SOURCES["native"]
        plat = "win-amd64" if IS_WIN else "linux-x86_64"
        asset = nb["assets"][plat]
        req = INSTALL / "requirements" / "export.txt"
        fp = asset["sha256"] + file_sha256(req)
        if self.done("export", fp) and venv_python(TP / "alembic" / ".venv").exists() and venv_python(TP / "fbx" / ".venv").exists():
            say(T("  已就绪", "  ready"))
            return
        zpath = DOWNLOADS / asset["file"]
        if self.args.native_dir:
            local = Path(self.args.native_dir) / asset["file"]
            if not local.exists():
                raise Abort(T(f"找不到 {local}", f"not found: {local}"))
            DOWNLOADS.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, zpath)
        urls = [m["url"].format(repo=nb["repo"], commit=nb["commit"], file=asset["file"]) for m in order_mirrors("github_release", self.region)]
        fetch_any(urls, zpath, asset["sha256"], asset.get("size"), label=asset["file"])
        for name in ("alembic", "fbx"):
            py = self.uv.venv(TP / name / ".venv", "3.12")
            self.uv.pip(py, ["-r", str(req)], self.uv.pypi())
            sp = site_packages(py)
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.namelist():
                    rel = Path(member)
                    if rel.is_absolute() or ".." in rel.parts:
                        raise Abort(f"unsafe path in {zpath.name}: {member}")
                zf.extractall(sp)
            if not IS_WIN:
                for p in sp.iterdir():
                    if p.suffix == ".so" or ".so." in p.name:
                        p.chmod(0o755)
            point_at_project(py)
        self.mark("export", fp)

    # -- checks
    def check_gpu_env(self, venv: Path, label: str) -> bool:
        py = venv_python(venv)
        if not py.exists():
            say(f"  {label}: " + T("未安装", "not installed"))
            return False
        code = "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
        say(f"  {label}: {out.stdout.strip() or out.stderr.strip()[-300:]}")
        return out.returncode == 0 and " True " in f" {out.stdout.strip()} "


# ------------------------------------------------------------------------------------------------ the wizard
def system_check() -> dict:
    info = {"os": platform.platform(), "gpus": [], "driver": ""}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                          text=True, timeout=20)
            for line in out.strip().splitlines():
                name, mem, drv = [x.strip() for x in line.split(",")]
                info["gpus"].append(f"{name} ({mem})")
                info["driver"] = drv
        except Exception:  # noqa: BLE001
            pass
    info["free"] = shutil.disk_usage(ROOT).free
    return info


def main() -> int:
    global LANG
    ap = argparse.ArgumentParser(description="ShotSpace installer")
    ap.add_argument("--uv", required=True, help=argparse.SUPPRESS)
    ap.add_argument("--lang", choices=["zh", "en"], default="")
    ap.add_argument("--yes", action="store_true", help="accept every default (no questions)")
    ap.add_argument("--region", choices=["auto", "cn", "global"], default="")
    ap.add_argument("--weights", default="", help="modelscope | huggingface | hf-mirror | a folder that already holds the weights")
    ap.add_argument("--objects", action="store_true", help="also install object matching (Linux)")
    ap.add_argument("--no-export", action="store_true", help="skip the Alembic / FBX export modules")
    ap.add_argument("--native-dir", default="", help="a folder holding the native export bundle (instead of downloading it)")
    ap.add_argument("--accept-licenses", action="store_true")
    ap.add_argument("--token-to-mirror", action="store_true", help="also send HF_TOKEN to hf-mirror.com (a third party)")
    ap.add_argument("--check", action="store_true", help="only check what is installed")
    args = ap.parse_args()

    if args.lang:
        LANG = args.lang
    else:
        loc = (os.environ.get("LANG") or os.environ.get("LC_ALL") or (locale.getlocale()[0] or "")).lower()
        LANG = "zh" if loc.startswith("zh") or "chinese" in loc else "en"
        if not args.yes:
            LANG = {"1": "zh", "2": "en"}[ask("语言 / Language: 1 中文  2 English", "1" if LANG == "zh" else "2", ["1", "2"])]
    y = args.yes
    for d in (TOOLS / "tmp", DOWNLOADS):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(TOOLS / "tmp")
    tempfile.tempdir = str(TOOLS / "tmp")

    head(T("入镜 · ShotSpace 安装向导", "ShotSpace installer"))
    say(T(f"安装位置：{ROOT}\n所有内容都放在这个文件夹里，不改动系统。中途断了重新运行即可继续。",
          f"Location: {ROOT}\nEverything stays inside this folder; nothing system-wide is changed. Re-run to continue."))
    info = system_check()
    say(T(f"系统：{info['os']}", f"System: {info['os']}"))
    say(T(f"显卡：{', '.join(info['gpus']) or '未检测到 NVIDIA 显卡'}", f"GPU: {', '.join(info['gpus']) or 'no NVIDIA GPU found'}"))
    if info["driver"]:
        say(T(f"驱动：{info['driver']}", f"Driver: {info['driver']}"))
    say(T(f"剩余空间：{human(info['free'])}", f"Free disk: {human(info['free'])}"))

    if args.check:
        inst = Installer(args)
        head(T("检查", "Check"))
        ok1 = inst.check_gpu_env(TP / "sam_3d_body" / ".venv", "SAM 3D Body")
        ok2 = inst.check_gpu_env(TP / "sam_3d_objects" / ".venv", "SAM 3.1")
        return 0 if ok1 and ok2 else 1

    if not info["gpus"]:
        say(T("警告：没有检测到 NVIDIA 显卡或驱动。解算需要一块 24 GB 显存左右的 NVIDIA 显卡（驱动 570 以上）。",
              "Warning: no NVIDIA GPU / driver found. Solving needs an NVIDIA GPU with about 24 GB (driver 570+)."))
        if not ask_yes(T("仍然继续安装？", "Continue anyway?"), False, y):
            return 1
    else:
        try:
            if int(info["driver"].split(".")[0]) < 570:
                say(T("警告：驱动版本低于 570，CUDA 12.8 需要 570 以上，请先升级驱动。",
                      "Warning: driver older than 570; CUDA 12.8 needs 570+. Please update the driver."))
        except ValueError:
            pass

    # ---- choices
    head(T("1/3 选择", "1/3 Choices"))
    region = args.region or ("auto" if y else {"1": "auto", "2": "cn", "3": "global"}[ask(
        T("下载源：1 自动测速选最快  2 国内镜像优先  3 官方源优先", "Download sources: 1 measure and pick the fastest  2 mainland China mirrors first  3 official first"),
        "1", ["1", "2", "3"])])
    args.region = region

    export = not args.no_export
    if not args.no_export and not y:
        export = ask_yes(T("安装 ABC / FBX 导出（推荐）？USD 导出总是可用。", "Install ABC / FBX export (recommended)? USD export is always available."), True)
    objects = args.objects
    if IS_WIN:
        if objects:
            say(T("物体匹配只支持 Linux，Windows 上跳过。", "Object matching is Linux only; skipped on Windows."))
        objects = False
    elif not y and not args.objects:
        objects = ask_yes(T("安装物体匹配（可选，模型约 15 GB）？", "Install object matching (optional, models about 15 GB)?"), False)

    if not args.weights:
        if y:
            args.weights = "modelscope" if region != "global" else "huggingface"
        else:
            pick = ask(T("模型权重从哪里下载：\n  1 ModelScope 魔搭（国内快，不需要账号）\n  2 HuggingFace（需要账号、在模型页同意许可并设置 HF_TOKEN）\n  3 hf-mirror.com（HuggingFace 的国内镜像）\n  4 本机已有的文件夹",
                         "Where to download the model weights from:\n  1 ModelScope (no account needed)\n  2 HuggingFace (needs an account, the licence accepted on each model page, HF_TOKEN)\n  3 hf-mirror.com (a HuggingFace mirror)\n  4 a folder on this computer that already holds them"),
                       "1" if region != "global" else "2", ["1", "2", "3", "4"])
            if pick == "4":
                args.weights = ask(T("文件夹路径", "Folder"), str(TP), None)
            else:
                args.weights = {"1": "modelscope", "2": "huggingface", "3": "hf-mirror"}[pick]
    if args.weights in ("huggingface", "hf-mirror") and not os.environ.get("HF_TOKEN") and not y:
        import getpass

        say(T("SAM 3D Body 和 SAM 3.1 在 HuggingFace 上需要先在模型页申请并同意许可：\n  https://huggingface.co/facebook/sam-3d-body-dinov3\n  https://huggingface.co/facebook/sam3.1\n然后在 https://huggingface.co/settings/tokens 创建一个只读 token。",
              "SAM 3D Body and SAM 3.1 are gated on HuggingFace: request access and accept the licence on\n  https://huggingface.co/facebook/sam-3d-body-dinov3\n  https://huggingface.co/facebook/sam3.1\nthen create a read token at https://huggingface.co/settings/tokens."))
        tok = getpass.getpass(T("粘贴 HuggingFace token（不显示，不保存；直接回车则改用 ModelScope）：", "Paste the HuggingFace token (hidden, not stored; Enter = use ModelScope instead): ")).strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
        else:
            args.weights = "modelscope"
    if args.weights == "hf-mirror" and os.environ.get("HF_TOKEN") and not args.token_to_mirror and not y:
        args.token_to_mirror = ask_yes(T("要把 token 发给 hf-mirror.com（第三方网站）吗？不发则受限模型改从其他源下载。",
                                         "Send your token to hf-mirror.com (a third party)? If not, gated models come from another source."), False)

    # ---- licences
    head(T("2/3 许可", "2/3 Licences"))
    say(T("ShotSpace 本身是 MIT 许可。它使用的模型有各自的许可，安装即表示你同意：",
          "ShotSpace itself is MIT licensed. The models it uses come with their own licences; installing means you accept them:"))
    for lic in SOURCES["licences"]:
        if lic.get("objects") and not objects:
            continue
        say(f"  - {lic['name']}: {lic['url']}")
    if not (args.accept_licenses or ask_yes(T("同意以上许可并继续？", "Accept these licences and continue?"), True, y)):
        return 1

    need = 25 * 2**30 + (25 * 2**30 if objects else 0)
    if info["free"] < need:
        say(T(f"警告：剩余空间 {human(info['free'])}，建议至少 {human(need)}。", f"Warning: {human(info['free'])} free, at least {human(need)} recommended."))
        if not ask_yes(T("仍然继续？", "Continue anyway?"), False, y):
            return 1

    # ---- work
    inst = Installer(args)
    steps = [
        (T("ShotSpace 服务环境", "ShotSpace service environment"), inst.main_env),
        (T("SAM 3D Body 源码", "SAM 3D Body source"), lambda: [inst.source("sam-3d-body"), inst.source("dinov3")]),
        (T("SAM 3.1 源码（含 ShotSpace 的修改）", "SAM 3.1 source (with ShotSpace's patch)"), lambda: inst.source("sam3")),
        (T("人体解算环境（SAM 3D Body）", "Body worker environment (SAM 3D Body)"),
         lambda: inst.gpu_env("body", TP / "sam_3d_body" / ".venv", [INSTALL / "requirements" / "body.txt"])),
        (T("人物跟踪环境（SAM 3.1）", "Tracking worker environment (SAM 3.1)"),
         lambda: inst.gpu_env("tracker", TP / "sam_3d_objects" / ".venv",
                              [INSTALL / "requirements" / "tracker.txt"] + ([INSTALL / "requirements" / "objects.txt"] if objects else []))),
        (T("模型：SAM 3D Body（2.8 GB）", "Model: SAM 3D Body (2.8 GB)"), lambda: inst.weights("sam-3d-body-dinov3")),
        (T("模型：SAM 3.1（3.5 GB）", "Model: SAM 3.1 (3.5 GB)"), lambda: inst.weights("sam3.1")),
    ]
    if export:
        steps.append((T("ABC / FBX 导出模块", "ABC / FBX export modules"), inst.export_envs))
    if objects:
        steps += [
            (T("物体匹配源码", "Object matching source"), lambda: [inst.source(n) for n in ("sam-3d-objects", "moge", "triposr")]),
            (T("物体匹配模型（约 15 GB）", "Object matching models (about 15 GB)"),
             lambda: [inst.weights(n) for n in ("sam-3d-objects", "moge-vitl", "triposr", "dino-vitb16")]),
        ]
    head(T("3/3 安装", "3/3 Installing"))
    t0 = time.time()
    optional = {inst.export_envs}
    skipped = []
    for i, (label, fn) in enumerate(steps, 1):
        say(f"\n[{i}/{len(steps)}] {label}")
        if fn in optional:
            try:
                fn()
            except Abort as exc:   # USD export works without it: finish the rest, say what is missing
                say(T(f"  没有装上：{exc}", f"  not installed: {exc}"))
                skipped.append(label)
            continue
        fn()

    head(T("检查", "Check"))
    ok = inst.check_gpu_env(TP / "sam_3d_body" / ".venv", "SAM 3D Body")
    ok = inst.check_gpu_env(TP / "sam_3d_objects" / ".venv", "SAM 3.1") and ok
    head(T(f"完成（用时 {int(time.time() - t0) // 60} 分钟）", f"Done ({int(time.time() - t0) // 60} min)"))
    start = "shotspace.bat start" if IS_WIN else "./shotspace.sh start"
    say(T(f"启动服务：{start}\n然后用 Chrome 打开 http://<这台电脑的地址>:8765（本机：http://127.0.0.1:8765）",
          f"Start the service: {start}\nthen open http://<this computer>:8765 in Chrome (here: http://127.0.0.1:8765)"))
    for label in skipped:
        say(T(f"注意：{label} 没有装上（USD 导出不受影响）。网络恢复后重新运行安装程序即可补上。",
              f"Note: {label} was not installed (USD export still works). Run the installer again later to add it."))
    if not ok:
        say(T("注意：显卡检查没有通过（见上面），解算需要可用的 NVIDIA 显卡。", "Note: the GPU check did not pass (see above); solving needs a working NVIDIA GPU."))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Abort as exc:
        say()
        say(T("安装中止：", "Setup stopped: ") + str(exc))
        say(T("修复后重新运行安装程序即可从这里继续。", "Run the installer again to continue from here."))
        sys.exit(2)
    except KeyboardInterrupt:
        say()
        say(T("已中断。重新运行即可继续。", "Interrupted. Run again to continue."))
        sys.exit(130)
