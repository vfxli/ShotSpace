#!/usr/bin/env bash
# ShotSpace installer (Linux).  Run:  ./setup.sh
#
# Everything goes inside this folder: the package manager (uv), Python, the environments, the models and every
# cache live under ./.tools and ./third_party.  Nothing is installed system-wide, nothing outside the folder is
# changed.  Re-running continues where the last run stopped.  Options are passed on to install/wizard.py
# (./setup.sh --help).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${ROOT}/.tools"
UV_DIR="${TOOLS}/uv"
UV="${UV_DIR}/uv"

UV_VERSION="0.13.7"
UV_WHEEL_PATH="packages/b9/93/0e053a2ca4523ee9825ccbedeb7121470d0a188933f11688b6d90d7f5c08/uv-0.13.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_SHA256="2c9bdc4a872b60a93c5f26c0791089ea11585a4f9d3b3c017d78eb8487505e25"
UV_WHEEL_HOSTS=(
  "https://files.pythonhosted.org"
  "https://mirrors.aliyun.com/pypi"
  "https://pypi.tuna.tsinghua.edu.cn"
  "https://mirrors.ustc.edu.cn/pypi"
)
PYTHON_MIRRORS=(
  ""                                                          # uv's default (GitHub)
  "https://registry.npmmirror.com/-/binary/python-build-standalone"
)

say() { printf '%s\n' "$*"; }

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  say "ShotSpace setup.sh supports Linux x86_64 only (on Windows run setup.bat)."
  exit 1
fi
command -v curl >/dev/null 2>&1 || { say "curl is required / 需要 curl"; exit 1; }

mkdir -p "${UV_DIR}" "${TOOLS}/tmp"
export UV_CACHE_DIR="${TOOLS}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${TOOLS}/python"
export UV_MANAGED_PYTHON=1
export UV_NO_CONFIG=1
export TMPDIR="${TOOLS}/tmp"

extract_uv() {   # wheel -> uv binary (a wheel is a zip)
  local whl="$1"
  if command -v unzip >/dev/null 2>&1; then
    unzip -o -q -j "${whl}" "uv-${UV_VERSION}.data/scripts/uv" -d "${UV_DIR}"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "${whl}" "${UV_DIR}" "${UV_VERSION}" <<'PY'
import sys, zipfile, pathlib
whl, out, ver = sys.argv[1:4]
with zipfile.ZipFile(whl) as z:
    pathlib.Path(out, "uv").write_bytes(z.read(f"uv-{ver}.data/scripts/uv"))
PY
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xOf "${whl}" "uv-${UV_VERSION}.data/scripts/uv" > "${UV_DIR}/uv"
  else
    return 1
  fi
  chmod +x "${UV}"
}

if [[ ! -x "${UV}" ]] || [[ "$("${UV}" --version 2>/dev/null | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  say "==> uv ${UV_VERSION}"
  whl="${TOOLS}/tmp/uv-${UV_VERSION}.whl"
  ok=0
  # fastest source first: time a tiny piece of the file from each
  ranked="$(for host in "${UV_WHEEL_HOSTS[@]}"; do
      t="$(curl -s -o /dev/null -r 0-65535 -w '%{http_code} %{time_total}' --connect-timeout 8 --max-time 15 "${host}/${UV_WHEEL_PATH}" 2>/dev/null || true)"
      case "${t}" in 2*) printf '%s %s\n' "${t#* }" "${host}" ;; *) printf '999 %s\n' "${host}" ;; esac
    done | sort -n | awk '{print $2}')"
  for host in ${ranked}; do
    say "    ${host}"
    # a source slower than 100 KB/s for 20 s is dropped for the next one
    if curl -fL --retry 2 --connect-timeout 15 --speed-limit 102400 --speed-time 20 --max-time 900 -o "${whl}" "${host}/${UV_WHEEL_PATH}" 2>/dev/null; then
      if [[ "$(sha256sum "${whl}" | awk '{print $1}')" == "${UV_WHEEL_SHA256}" ]]; then
        ok=1; break
      fi
      say "    checksum mismatch, trying the next source / 校验不一致，换下一个源"
    fi
  done
  if [[ "${ok}" != 1 ]]; then
    say "Could not download uv from any source. / 所有下载源都失败了，请检查网络后重新运行 ./setup.sh"
    exit 1
  fi
  extract_uv "${whl}" || { say "Need one of: unzip, python3, bsdtar / 需要 unzip、python3 或 bsdtar 之一"; exit 1; }
  rm -f "${whl}"
fi

# Python for the wizard (the environments get theirs from the wizard)
if ! "${UV}" python find 3.12 >/dev/null 2>&1; then
  say "==> Python 3.12"
  ok=0
  # order: as --region says, else the source that answers first
  region=""
  prev=""
  for a in "$@"; do
    [[ "${prev}" == "--region" ]] && region="${a}"
    [[ "${a}" == --region=* ]] && region="${a#--region=}"
    prev="${a}"
  done
  t_gh="$(curl -s -o /dev/null -w '%{time_total}' --connect-timeout 8 --max-time 15 https://github.com/astral-sh/python-build-standalone/releases 2>/dev/null || echo 999)"
  t_cn="$(curl -s -o /dev/null -w '%{time_total}' --connect-timeout 8 --max-time 15 "${PYTHON_MIRRORS[1]}/" 2>/dev/null || echo 999)"
  if [[ "${region}" == "cn" ]] || { [[ "${region}" != "global" ]] && awk "BEGIN{exit !(${t_cn} < ${t_gh})}"; }; then
    PYTHON_MIRRORS=("${PYTHON_MIRRORS[1]}" "")
  fi
  for m in "${PYTHON_MIRRORS[@]}"; do
    if [[ -n "${m}" ]]; then say "    ${m}"; args=(--mirror "${m}"); else say "    github.com"; args=(); fi
    if "${UV}" python install 3.12 --no-bin --no-registry "${args[@]}"; then ok=1; break; fi
  done
  [[ "${ok}" == 1 ]] || { say "Could not download Python. / Python 下载失败，请检查网络后重新运行 ./setup.sh"; exit 1; }
fi
PY="$("${UV}" python find 3.12)"

exec "${PY}" -I "${ROOT}/install/wizard.py" --uv "${UV}" "$@"
