"""Persistent server configuration & `/admin` authentication for ShotSpace.

Stores `<PROJECT_ROOT>/config/server_config.json`:
  - `initialized`: bool (whether first-run port + admin password setup has completed)
  - `host`: str (default "0.0.0.0")
  - `port`: int (default 8765)
  - `password_salt`: hex str
  - `password_hash`: PBKDF2-HMAC-SHA256 hex str
  - `secret_key`: 32-byte random hex str for signing stateless admin tokens
  - `disabled_gpus`: list[int] of GPU indices turned OFF in `/admin`
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_PORT, PROJECT_ROOT

PBKDF2_ITERATIONS = 200_000
TOKEN_TTL_SECONDS = 86400 * 7  # 7 days


def get_config_file() -> Path:
    override = os.environ.get("SHOTSPACE_CONFIG_FILE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "config" / "server_config.json"


def _hash_password(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS,
    )
    return dk.hex()


@dataclass
class ServerConfig:
    initialized: bool = False
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    password_salt: str = ""
    password_hash: str = ""
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    disabled_gpus: list[int] = field(default_factory=list)
    retention_days: int = 30  # Default 30 days (1 month) retention for solved task data
    cache_dir: str = ""  # Optional custom workspace/cache directory (e.g. mounted studio NAS path)
    updated_at: str = ""


def validate_and_prepare_cache_dir(raw_path: str) -> str:
    """Validate and create a custom cache/workspace directory (or return `""` for default).
    Ensures the directory is writable and never points at `/` or ShotSpace code/weights folders."""
    cleaned = (raw_path or "").strip()
    if not cleaned:
        return ""
    target = Path(os.path.abspath(Path(cleaned).expanduser()))
    root_resolved = PROJECT_ROOT.resolve()
    forbidden = {
        Path("/"),
        root_resolved,
        (root_resolved / "shotspace").resolve(),
        (root_resolved / "third_party").resolve(),
        (root_resolved / "webui").resolve(),
        (root_resolved / "config").resolve(),
        (root_resolved / "secrets").resolve(),
    }
    norm = os.path.normcase(str(target))
    if target in forbidden or target.parent == target or norm in {os.path.normcase(str(f)) for f in forbidden}:
        # a filesystem root ("/", or a drive such as "E:\\" on Windows) is never a cache folder
        raise ValueError(f"缓存路径不能直接指向程序代码或权重受保护目录：{target}")
    for prot in (
        (root_resolved / "shotspace").resolve(),
        (root_resolved / "third_party").resolve(),
        (root_resolved / "webui").resolve(),
        (root_resolved / "secrets").resolve(),
    ):
        if norm.startswith(os.path.normcase(str(prot)) + os.sep):
            raise ValueError(f"缓存路径不能位于受保护子目录内：{target}")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"无法创建缓存目录 {target}：{exc}") from exc
    if not target.is_dir():
        raise ValueError(f"目标缓存路径不是有效目录：{target}")
    probe = target / f".shotspace_write_probe_{os.getpid()}_{secrets.token_hex(4)}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError(f"目标缓存目录不可写（请检查服务器挂载权限）：{target} ({exc})") from exc
    default_ws = (root_resolved / "workspaces").resolve()
    if target == default_ws and not os.environ.get("SHOTSPACE_WORKSPACES", "").strip():
        return ""
    return str(target)


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_config_file()
        self._lock = threading.RLock()

    def load(self) -> ServerConfig:
        with self._lock:
            if not self.path.is_file():
                return ServerConfig()
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                return ServerConfig(
                    initialized=bool(raw.get("initialized", False)),
                    host=str(raw.get("host", "0.0.0.0")),
                    port=int(raw.get("port", DEFAULT_PORT)),
                    password_salt=str(raw.get("password_salt", "")),
                    password_hash=str(raw.get("password_hash", "")),
                    secret_key=str(raw.get("secret_key") or secrets.token_hex(32)),
                    disabled_gpus=[int(x) for x in (raw.get("disabled_gpus") or [])],
                    retention_days=int(raw.get("retention_days", 30)),
                    cache_dir=str(raw.get("cache_dir") or "").strip(),
                    updated_at=str(raw.get("updated_at", "")),
                )
            except Exception:
                return ServerConfig()

    def save(self, cfg: ServerConfig) -> ServerConfig:
        with self._lock:
            cfg.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            payload = {
                "initialized": cfg.initialized,
                "host": cfg.host,
                "port": int(cfg.port),
                "password_salt": cfg.password_salt,
                "password_hash": cfg.password_hash,
                "secret_key": cfg.secret_key,
                "disabled_gpus": sorted({int(i) for i in cfg.disabled_gpus}),
                "retention_days": int(cfg.retention_days),
                "cache_dir": (cfg.cache_dir or "").strip(),
                "updated_at": cfg.updated_at,
            }
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)
            return cfg

    def initialize(
        self,
        port: int,
        password: str,
        host: str = "0.0.0.0",
        cache_dir: str = "",
    ) -> ServerConfig:
        if len(password) < 4:
            raise ValueError("后台密码长度不能少于 4 位")
        if not (1 <= int(port) <= 65535):
            raise ValueError("端口号必须在 1–65535 之间")
        prepared_cache = validate_and_prepare_cache_dir(cache_dir)
        with self._lock:
            cfg = self.load()
            salt = secrets.token_hex(16)
            cfg.initialized = True
            cfg.host = host
            cfg.port = int(port)
            cfg.password_salt = salt
            cfg.password_hash = _hash_password(password, salt)
            if cache_dir is not None:
                cfg.cache_dir = prepared_cache
            if not cfg.secret_key:
                cfg.secret_key = secrets.token_hex(32)
            return self.save(cfg)

    def verify_password(self, password: str) -> bool:
        cfg = self.load()
        if not cfg.initialized or not cfg.password_salt or not cfg.password_hash:
            return False
        cand = _hash_password(password, cfg.password_salt)
        return hmac.compare_digest(cand, cfg.password_hash)

    def change_password(self, old_password: str, new_password: str) -> ServerConfig:
        if len(new_password) < 4:
            raise ValueError("新密码长度不能少于 4 位")
        with self._lock:
            if not self.verify_password(old_password):
                raise ValueError("当前密码错误")
            cfg = self.load()
            salt = secrets.token_hex(16)
            cfg.password_salt = salt
            cfg.password_hash = _hash_password(new_password, salt)
            return self.save(cfg)

    def force_set_password(self, new_password: str) -> ServerConfig:
        if len(new_password) < 4:
            raise ValueError("新密码长度不能少于 4 位")
        with self._lock:
            cfg = self.load()
            salt = secrets.token_hex(16)
            cfg.initialized = True
            cfg.password_salt = salt
            cfg.password_hash = _hash_password(new_password, salt)
            return self.save(cfg)

    def set_port(self, port: int) -> ServerConfig:
        if not (1 <= int(port) <= 65535):
            raise ValueError("端口号必须在 1–65535 之间")
        with self._lock:
            cfg = self.load()
            cfg.port = int(port)
            return self.save(cfg)

    def set_disabled_gpus(self, disabled: list[int]) -> ServerConfig:
        with self._lock:
            cfg = self.load()
            cfg.disabled_gpus = sorted({int(x) for x in disabled})
            return self.save(cfg)

    def set_retention_days(self, days: int) -> ServerConfig:
        if int(days) < 0 or int(days) > 3650:
            raise ValueError("数据保留天数需在 0–3650 天之间（0 表示永久保留）")
        with self._lock:
            cfg = self.load()
            cfg.retention_days = int(days)
            return self.save(cfg)

    def set_cache_dir(self, cache_dir: str) -> ServerConfig:
        prepared = validate_and_prepare_cache_dir(cache_dir)
        with self._lock:
            cfg = self.load()
            cfg.cache_dir = prepared
            return self.save(cfg)

    def _signing_key(self, cfg: ServerConfig) -> bytes:
        # Mix password_hash into signing key so changing password immediately revokes old tokens!
        return f"{cfg.secret_key}:{cfg.password_hash[:24]}".encode("utf-8")

    def create_token(self) -> str:
        cfg = self.load()
        exp = int(time.time()) + TOKEN_TTL_SECONDS
        body = json.dumps({"role": "admin", "exp": exp}, separators=(",", ":")).encode("utf-8")
        b64 = base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")
        sig = hmac.new(self._signing_key(cfg), b64.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{b64}.{sig}"

    def verify_token(self, token: str | None) -> bool:
        if not token or "." not in token:
            return False
        cfg = self.load()
        if not cfg.initialized:
            return False
        b64, _, sig = token.partition(".")
        expected = hmac.new(self._signing_key(cfg), b64.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        try:
            pad = "=" * ((-len(b64)) % 4)
            data = json.loads(base64.urlsafe_b64decode(b64 + pad).decode("utf-8"))
            return data.get("role") == "admin" and int(data.get("exp", 0)) > int(time.time())
        except Exception:
            return False
