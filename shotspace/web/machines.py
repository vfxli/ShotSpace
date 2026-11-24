"""Which computer a request comes from, and which computer a task belongs to (v0.2.12).

Every task belongs to the computer that created it, identified by its machine name and its address on the network.
A computer sees and opens only its own tasks (and the public ones, see below); the admin page sees all of them.

* The address is the one the request comes from (`app._client_ip`).  The machine name is looked up from the
  address on the server - by reverse DNS, and for Windows computers by asking the computer itself for its name
  (NetBIOS name query, UDP 137) - since a browser does not tell it.  Both lookups are short and cached; when neither
  answers, a computer is known by its address alone.
* Two records are the same computer when both have a name and the names are equal (the address may have changed),
  or - when a name is missing on either side - when the addresses are equal.  When a computer is recognised its
  record is completed: a name learnt later is added, an address that changed is updated.

Tasks made before v0.2.12 have no owner.  When the service starts, each of them is given to the computer the
service's own records name: its log (`shotspace.log`, `shotspace.log.old`: every request with the address it came
from) and the queue history (every job with the address that submitted it).  What cannot be decided with
certainty becomes an UNCLASSIFIED task ("未分类", stored as `{"public": True}`): every computer sees it, whoever it
belongs to moves it into their own list with one click, and the admin page gives it to a computer or deletes it.
Nothing is deleted or moved on disk by the upgrade.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def normalise_ip(raw: str) -> str:
    """'::ffff:10.0.0.5' -> '10.0.0.5'; anything unparsable is kept as it came."""
    text = str(raw or "").strip().strip("[]")
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return text
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return str(addr.ipv4_mapped)
    return str(addr)


def _short_name(name: str) -> str:
    return str(name or "").strip().split(".")[0].strip()


def _reverse_dns(ip: str) -> str:
    try:
        name = socket.gethostbyaddr(ip)[0]
    except (OSError, UnicodeError, ValueError):
        return ""
    short = _short_name(name)
    return "" if (not short or short == ip) else short


def _netbios_name(ip: str, timeout: float = 0.6, port: int = 137) -> str:
    """Ask a Windows computer for its own name (NetBIOS node status query).  '' when it does not answer."""
    try:
        if ipaddress.ip_address(ip).version != 4:
            return ""
    except ValueError:
        return ""
    tid = os.urandom(2)
    # header (id, flags 0, one question) + the name '*' in first-level encoding + type NBSTAT + class IN
    raw = b"*" + b"\x00" * 15
    encoded = bytes(v for c in raw for v in ((c >> 4) + 0x41, (c & 0x0F) + 0x41))
    query = tid + b"\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00" + b"\x20" + encoded + b"\x00" + b"\x00\x21\x00\x01"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sock.sendto(query, (ip, port))
        data, _addr = sock.recvfrom(4096)
    except OSError:
        return ""
    finally:
        sock.close()
    try:
        if data[:2] != tid or len(data) < 57:
            return ""
        pos = 12
        if data[pos] & 0xC0 == 0xC0:
            pos += 2
        else:
            while data[pos] != 0:
                pos += data[pos] + 1
            pos += 1
        pos += 2 + 2 + 4 + 2  # type, class, ttl, length
        count = data[pos]
        pos += 1
        for _ in range(count):
            entry = data[pos : pos + 18]
            pos += 18
            if len(entry) < 18:
                break
            name = entry[:15].decode("ascii", "replace").strip()
            suffix, flags = entry[15], int.from_bytes(entry[16:18], "big")
            if suffix == 0x00 and not (flags & 0x8000) and name:  # the computer's own (unique) name
                return name
    except (IndexError, ValueError):
        return ""
    return ""


def lookup_name(ip: str) -> str:
    """Machine name of an address: reverse DNS first, then the computer itself.  '' when unknown."""
    return _reverse_dns(ip) or _netbios_name(ip)


class MachineNames:
    """Names of the computers using the service, by address.  A name is looked up once (at most about a second, on
    a thread of its own so a resolver that hangs never holds a request up) and then kept for `ttl_s`; after that it
    is refreshed in the background while the old answer is still given.

    It also remembers which computers have opened the service and when last (`seen`, kept in `seen_file` so the
    admin page still shows them after a restart)."""

    def __init__(self, ttl_s: float = 600.0, wait_s: float = 1.2, seen_file: Path | None = None) -> None:
        self.ttl_s = float(ttl_s)
        self.wait_s = float(wait_s)
        self.resolver = lookup_name  # replaceable (tests)
        self._cache: dict[str, tuple[str, float]] = {}
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.seen_file = seen_file
        self.seen: dict[str, dict] = {}       # address -> {"ip", "name", "last_seen"}
        self._seen_saved = 0.0
        if seen_file is not None and seen_file.is_file():
            try:
                import json

                loaded = json.loads(seen_file.read_text(encoding="utf-8"))
                self.seen = {str(k): dict(v) for k, v in dict(loaded).items() if isinstance(v, dict)}
            except Exception:  # noqa: BLE001 - an unreadable record of visits is started afresh
                self.seen = {}

    def _note_seen(self, ip: str, name: str) -> None:
        now = time.time()
        with self._lock:
            cur = self.seen.get(ip)
            changed = cur is None or (name and cur.get("name") != name)
            self.seen[ip] = {"ip": ip, "name": name or (cur or {}).get("name", ""), "last_seen": now}
            due = changed or now - self._seen_saved > 60.0
            if due:
                self._seen_saved = now
                snapshot = {k: dict(v) for k, v in self.seen.items()}
        if due and self.seen_file is not None:
            try:
                import json

                tmp = self.seen_file.with_name(f".{self.seen_file.name}.tmp")
                tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(self.seen_file)
            except OSError:
                pass

    def _resolve(self, ip: str, done: threading.Event) -> None:
        try:
            name = str(self.resolver(ip) or "")
        except Exception:  # noqa: BLE001 - a failed lookup means: no name
            name = ""
        with self._lock:
            self._cache[ip] = (name, time.time())
            self._pending.pop(ip, None)
        done.set()

    def name_of(self, ip: str) -> str:
        ip = normalise_ip(ip)
        if not ip:
            return ""
        with self._lock:
            hit = self._cache.get(ip)
            event = self._pending.get(ip)
            if event is None and (hit is None or time.time() - hit[1] > self.ttl_s):
                event = threading.Event()
                self._pending[ip] = event
                threading.Thread(target=self._resolve, args=(ip, event), name="shotspace-machine-name", daemon=True).start()
        if hit is not None:
            return hit[0]
        if event is not None:
            event.wait(self.wait_s)
        with self._lock:
            hit = self._cache.get(ip)
        return hit[0] if hit is not None else ""

    def viewer(self, ip: str) -> dict:
        ip = normalise_ip(ip)
        out = {"ip": ip, "name": self.name_of(ip)}
        if ip:
            self._note_seen(ip, out["name"])
        return out


def same_machine(owner: dict | None, viewer: dict | None) -> bool:
    """Is `viewer` the computer recorded in `owner`?  Names decide when both are known, addresses otherwise."""
    if not owner or not viewer or owner.get("public"):
        return False
    own_name, view_name = str(owner.get("name") or "").lower(), str(viewer.get("name") or "").lower()
    if own_name and view_name:
        return own_name == view_name
    own_ip, view_ip = normalise_ip(owner.get("ip") or ""), normalise_ip(viewer.get("ip") or "")
    return bool(own_ip) and own_ip == view_ip


def machine_key(owner: dict | None) -> str:
    """One string per computer, for counting tasks by computer: its name when known, else its address."""
    if not owner or owner.get("public"):
        return ""
    name = str(owner.get("name") or "").strip().lower()
    return f"name:{name}" if name else f"ip:{normalise_ip(owner.get('ip') or '')}"


def machine_label(owner: dict | None) -> str:
    """How a computer is shown: 'WS-023 (10.0.3.23)', or the address alone."""
    if not owner:
        return ""
    if owner.get("public"):
        return "未分类"
    name, ip = str(owner.get("name") or ""), str(owner.get("ip") or "")
    return f"{name} ({ip})" if (name and ip) else (name or ip)


# ------------------------------------------------------------------------------- tasks from before v0.2.12

_ACCESS = re.compile(r'^INFO:\s+(\S+):\d+\s+-\s+"([A-Z]+)\s+/api/shots/([A-Za-z0-9_\-]+)[^"\s]*\s+HTTP/[\d.]+"\s+(\d{3})')
_BANNER = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})\] Starting ")


@dataclass
class OwnerRecords:
    """What the service's own records say about who used which task."""

    log_start: float | None = None                                  # earliest moment the kept log covers
    requests: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # task -> [(address, method)] in order
    jobs: dict[str, list[str]] = field(default_factory=dict)        # task -> addresses that submitted its jobs
    log_files: list[str] = field(default_factory=list)


def read_owner_records(log_files: list[Path], history: list[dict]) -> OwnerRecords:
    """Read the service log (oldest file first) and the queue history.  Only requests that succeeded count."""
    rec = OwnerRecords()
    for path in log_files:
        if not path.is_file():
            continue
        rec.log_files.append(str(path))
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("INFO:"):
                        hit = _ACCESS.match(line)
                        if hit and hit.group(4).startswith("2"):
                            rec.requests.setdefault(hit.group(3), []).append((normalise_ip(hit.group(1)), hit.group(2)))
                    elif rec.log_start is None and line.startswith("["):
                        hit = _BANNER.match(line)
                        if hit:
                            try:
                                rec.log_start = datetime.strptime(f"{hit.group(1)} {hit.group(2)}", "%Y-%m-%d %H:%M:%S %z").timestamp()
                            except ValueError:
                                pass
        except OSError:
            continue
    for job in history or []:
        sid, ip = str(job.get("shot_id") or ""), normalise_ip(job.get("client_ip") or "")
        if sid and ip:
            rec.jobs.setdefault(sid, []).append(ip)
    return rec


def decide_owner(task_id: str, created_ts: float | None, rec: OwnerRecords) -> tuple[dict, str]:
    """Owner of a task made before v0.2.12 -> (owner record, the reason in words).

    1. The log goes back to before the task was created: the first computer that asked for the task is the one
       that created it (a new task is opened by its creator's page at once; nobody else knows its number yet).
    2. Otherwise: every computer the records show working on the task (log requests that change something, and
       the jobs of the queue history) is one and the same - that one.
    3. Otherwise - several computers, or no record at all - it becomes a public task."""
    reqs = rec.requests.get(task_id) or []
    if reqs and rec.log_start is not None and created_ts is not None and created_ts >= rec.log_start:
        return {"ip": reqs[0][0], "name": ""}, "日志记录了它的创建：第一个打开它的电脑"
    actors = {ip for ip, method in reqs if method not in ("GET", "HEAD")} | set(rec.jobs.get(task_id) or [])
    if len(actors) == 1:
        return {"ip": next(iter(actors)), "name": ""}, "所有操作记录都来自同一台电脑"
    if len(actors) > 1:
        return {"public": True}, f"操作记录来自 {len(actors)} 台电脑，无法确定"
    return {"public": True}, "没有找到操作记录"
