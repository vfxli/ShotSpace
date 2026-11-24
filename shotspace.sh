#!/usr/bin/env bash
# ==============================================================================
# ShotSpace Master Service, Venv Relocator, Hot-Update & Rollback Controller
# ==============================================================================
{
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_CWD="$(pwd)"
APP_NAME="ShotSpace"
PID_FILE="${APP_ROOT}/shotspace.pid"
LOG_FILE="${APP_ROOT}/shotspace.log"
LOG_MAX_BYTES=$((20 * 1024 * 1024))
VENV_DIR="${APP_ROOT}/.venv"
UPDATE_DIR="${APP_ROOT}/update"
APPLIED_DIR="${UPDATE_DIR}/applied"
BACKUP_DIR="${APP_ROOT}/backups"

DEFAULT_HOST="${SHOTSPACE_HOST:-0.0.0.0}"
DEFAULT_PORT="${SHOTSPACE_PORT:-8765}"

get_version() {
  local ver="0.1.0"
  if [[ -f "${APP_ROOT}/shotspace/__init__.py" ]]; then
    ver="$(sed -n 's/^__version__ *= *"\([^"]*\)".*/\1/p' "${APP_ROOT}/shotspace/__init__.py" | head -n 1)"
  fi
  printf '%s' "${ver:-0.1.0}"
}

rotate_log_if_needed() {
  if [[ -f "${LOG_FILE}" ]]; then
    local sz
    sz="$(stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)"
    if (( sz >= LOG_MAX_BYTES )); then
      mv -f "${LOG_FILE}" "${LOG_FILE}.old"
    fi
  fi
}

# ------------------------------------------------------------------------------
# Portable Python Virtualenv Relocator (Zero Internet / Zero Conda Required)
# ------------------------------------------------------------------------------
relocate_single_venv() {
  local vdir="$1"
  [[ -d "${vdir}" ]] || return 0

  # 1. If standard Python venv (`pyvenv.cfg` present), match & rewrite Python interpreter links
  if [[ -f "${vdir}/pyvenv.cfg" ]]; then
    local ver_mm=""
    ver_mm="$(sed -n 's/^version_info *= *\([0-9]\+\.[0-9]\+\).*/\1/p' "${vdir}/pyvenv.cfg" | head -n 1 || true)"

    local py_home=""
    local py_exe=""
    local cands=()
    if [[ -n "${ver_mm}" ]]; then
      cands+=(
        "${vdir}/cpython-${ver_mm}/bin"
        "${APP_ROOT}/third_party/_python/cpython-${ver_mm}-linux-x86_64-gnu/bin"
      )
    fi
    cands+=(
      "${vdir}/cpython-3.13/bin"
      "${vdir}/cpython-3.12/bin"
      "${vdir}/cpython-3.11/bin"
      "${APP_ROOT}/third_party/_python/cpython-3.13-linux-x86_64-gnu/bin"
      "${APP_ROOT}/third_party/_python/cpython-3.12-linux-x86_64-gnu/bin"
      "${APP_ROOT}/third_party/_python/cpython-3.11-linux-x86_64-gnu/bin"
    )

    for cand in "${cands[@]}"; do
      if [[ -n "${ver_mm}" && -x "${cand}/python${ver_mm}" ]]; then
        py_home="${cand}"
        py_exe="${cand}/python${ver_mm}"
        break
      elif [[ -x "${cand}/python3" ]]; then
        py_home="${cand}"
        py_exe="${cand}/python3"
        break
      fi
    done

    if [[ -n "${py_home}" && -n "${py_exe}" ]]; then
      sed -i \
        -e "s|^home *=.*|home = ${py_home}|" \
        -e "s|^executable *=.*|executable = ${py_exe}|" \
        -e "s|^command *=.*|command = ${py_exe} -m venv ${vdir}|" \
        "${vdir}/pyvenv.cfg"
      if [[ -L "${vdir}/bin/python" || ! -e "${vdir}/bin/python" ]]; then
        ln -sfn "${py_exe}" "${vdir}/bin/python"
      fi
      ln -sfn "python" "${vdir}/bin/python3"
      [[ -n "${ver_mm}" ]] && ln -sfn "python" "${vdir}/bin/python${ver_mm}"
    fi
  fi

  # 2. Rewrite editable .pth files so package imports always resolve to APP_ROOT
  while IFS= read -r -d '' pth_file; do
    if grep -qE 'ShotSpace|^/' "${pth_file}" 2>/dev/null; then
      printf '%s\n' "${APP_ROOT}" > "${pth_file}"
    fi
  done < <(find "${vdir}/lib" -maxdepth 3 -name "*shotspace*.pth" -print0 2>/dev/null || true)

  # Ensure shotspace.pth exists in site-packages
  local sp_dir
  sp_dir="$(find "${vdir}/lib" -maxdepth 3 -type d -name "site-packages" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${sp_dir}" ]]; then
    printf '%s\n' "${APP_ROOT}" > "${sp_dir}/shotspace_root.pth"
  fi

  # 3. Rewrite console-script shebangs in bin/* to current venv python
  if [[ -d "${vdir}/bin" ]]; then
    for script in "${vdir}/bin/"*; do
      [[ -f "${script}" && ! -L "${script}" ]] || continue
      if head -c 2 "${script}" 2>/dev/null | grep -q '^#!'; then
        if head -n 1 "${script}" 2>/dev/null | grep -q 'python'; then
          sed -i "1s|^#!.*python.*|#!${vdir}/bin/python|" "${script}" || true
        fi
      fi
    done
  fi
}

relocate_venv() {
  # Repair relative symlinks whose target had '${APP_NAME}/' prepended by GNU tar --transform
  local scan_dirs=()
  [[ -d "${VENV_DIR}" ]] && scan_dirs+=("${VENV_DIR}")
  [[ -d "${APP_ROOT}/third_party" ]] && scan_dirs+=("${APP_ROOT}/third_party")
  if (( ${#scan_dirs[@]} > 0 )); then
    while IFS= read -r -d '' lnk; do
      local tgt
      tgt="$(readlink "${lnk}" || true)"
      if [[ "${tgt}" == "${APP_NAME}/"* ]]; then
        ln -sfn "${tgt#${APP_NAME}/}" "${lnk}"
      fi
    done < <(find "${scan_dirs[@]}" -type l -lname "${APP_NAME}/*" -print0 2>/dev/null || true)
  fi

  relocate_single_venv "${VENV_DIR}"
  for sub_v in "${APP_ROOT}"/third_party/*/.venv; do
    [[ -d "${sub_v}" ]] && relocate_single_venv "${sub_v}"
  done
  if [[ -d "${APP_ROOT}/third_party" ]]; then
    PYTHONPATH="${APP_ROOT}" "$(python_bin)" -c "from shotspace.elf_compat import heal_runtime_elf_tree; heal_runtime_elf_tree('${APP_ROOT}/third_party')" 2>/dev/null || true
  fi
}

python_bin() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    printf '%s' "${VENV_DIR}/bin/python"
  else
    command -v python3
  fi
}

run_import_precheck() {
  local py
  py="$(python_bin)"
  PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" -c "
import sys
import shotspace
import shotspace.config
import shotspace.core.camera_space
import shotspace.core.rigid_object
import shotspace.io
import shotspace.web.app
print(f'[PRECHECK OK] ShotSpace v{shotspace.__version__} ({sys.executable})')
"
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local cmdline
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${APP_ROOT}"* || "${cmdline}" == *"shotspace"* ]]
}

# ------------------------------------------------------------------------------
# Precision Stop (Strictly Scoped to APP_ROOT Only — Never Kills Other Instances)
# ------------------------------------------------------------------------------
cmd_stop() {
  local stopped=0
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if is_running_pid "${pid}"; then
      echo "[STOP] Stopping ${APP_NAME} (PID ${pid})..."
      kill "${pid}" 2>/dev/null || true
      for _ in {1..25}; do
        is_running_pid "${pid}" || break
        sleep 0.2
      done
      if is_running_pid "${pid}"; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
      stopped=1
    fi
    rm -f "${PID_FILE}"
  fi

  # Fallback: strictly match exact APP_ROOT serve daemon only (never match shotspace.sh itself!)
  local pattern="${APP_ROOT}/.*(shotspace\.cli[[:space:]]+serve|bin/shotspace[[:space:]]+serve)"
  if pgrep -f "${pattern}" >/dev/null 2>&1; then
    pkill -f "${pattern}" 2>/dev/null || true
    sleep 0.5
    stopped=1
  fi

  if (( stopped == 1 )); then
    echo "[STOP] ${APP_NAME} stopped cleanly."
  else
    echo "[STOP] No running ${APP_NAME} process under ${APP_ROOT}."
  fi
}

ensure_first_run_config() {
  local req_host="$1"
  local req_port="$2"
  local req_pwd="$3"
  local py
  py="$(python_bin)"

  local is_init
  is_init="$(PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" -c '
from shotspace.server_config import ConfigStore
c = ConfigStore().load()
print("1" if (c.initialized and c.password_hash) else "0")
')"

  if [[ "${is_init}" == "1" ]]; then
    return 0
  fi

  echo "=============================================================================="
  echo "  入镜 · ShotSpace 首次运行初始化配置（监听端口与 /admin 后台管理密码）"
  echo "=============================================================================="

  local chosen_port="${req_port:-8765}"
  local chosen_pwd="${req_pwd}"

  if [[ -z "${chosen_pwd}" ]]; then
    local tty_in=""
    if [[ -t 0 ]]; then
      tty_in="/dev/stdin"
    elif [[ -r /dev/tty ]] && (: < /dev/tty) 2>/dev/null; then
      tty_in="/dev/tty"
    else
      echo "[ERROR] 首次运行必须配置监听端口与 /admin 后台管理密码！" >&2
      echo "        请在交互式终端直接运行: ./shotspace.sh start" >&2
      return 1
    fi

    while true; do
      local input_p=""
      printf "请输入 Web 服务监听端口 [默认 %s]: " "${chosen_port}"
      IFS= read -r input_p < "${tty_in}" || return 1
      input_p="${input_p//[[:space:]]/}"
      [[ -z "${input_p}" ]] && input_p="${chosen_port}"
      if [[ "${input_p}" =~ ^[0-9]+$ ]] && (( input_p >= 1 && input_p <= 65535 )); then
        chosen_port="${input_p}"
        break
      fi
      echo "[提示] 端口号必须为 1–65535 之间的整数，请重新输入。"
    done

    while true; do
      local p1="" p2=""
      printf "请设置 /admin 后台管理密码（至少 4 位，输入不回显）: "
      IFS= read -r -s p1 < "${tty_in}" || return 1
      echo ""
      if (( ${#p1} < 4 )); then
        echo "[提示] 密码长度不能少于 4 位，请重新输入。"
        continue
      fi
      printf "请再次输入确认后台管理密码: "
      IFS= read -r -s p2 < "${tty_in}" || return 1
      echo ""
      if [[ "${p1}" != "${p2}" ]]; then
        echo "[提示] 两次输入的密码不一致，请重新输入。"
        continue
      fi
      chosen_pwd="${p1}"
      break
    done
  fi

  printf '%s' "${chosen_pwd}" | PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${py}" -c '
import sys
from shotspace.server_config import ConfigStore
pwd = sys.stdin.read()
host = sys.argv[1]
port = int(sys.argv[2])
cfg = ConfigStore().initialize(port=port, password=pwd, host=host)
print(f"[OK] 首次运行配置已加密保存 -> {ConfigStore().path} (监听端口: {cfg.port})")
' "${req_host}" "${chosen_port}"
}

cmd_start() {
  local cli_host=""
  local cli_port="${SHOTSPACE_PORT:-}"
  local cli_pwd="${SHOTSPACE_ADMIN_PASSWORD:-}"
  local init_only=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host) cli_host="$2"; shift 2 ;;
      --port|-p) cli_port="$2"; shift 2 ;;
      --admin-password) cli_pwd="$2"; shift 2 ;;
      --init-only) init_only=1; shift ;;
      *) shift ;;
    esac
  done

  relocate_venv

  # Enforce first-run port + /admin password configuration before daemonizing
  ensure_first_run_config "${cli_host:-0.0.0.0}" "${cli_port}" "${cli_pwd}"
  if (( init_only == 1 )); then
    return 0
  fi

  local py cfg_line saved_host saved_port
  py="$(python_bin)"
  cfg_line="$(PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" -c '
from shotspace.server_config import ConfigStore
c = ConfigStore().load()
h = c.host if c.host else "0.0.0.0"
p = c.port if c.port else 8765
print(h + "|" + str(p))
')"
  saved_host="${cfg_line%%|*}"
  saved_port="${cfg_line##*|}"

  local host="${cli_host:-${saved_host:-0.0.0.0}}"
  local port="${cli_port:-${saved_port:-8765}}"

  if [[ -f "${PID_FILE}" ]]; then
    local cur_pid
    cur_pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if is_running_pid "${cur_pid}"; then
      echo "[START] ${APP_NAME} 已在运行中 (PID ${cur_pid}) -> http://${host}:${port}/"
      return 0
    fi
    rm -f "${PID_FILE}"
  fi

  rotate_log_if_needed

  local ver ts
  ver="$(get_version)"
  ts="$(date '+%Y-%m-%d %H:%M:%S %z')"

  {
    echo "=============================================================================="
    echo "[${ts}] Starting ${APP_NAME} v${ver} | root=${APP_ROOT} | listen=${host}:${port}"
    echo "=============================================================================="
  } >> "${LOG_FILE}"

  nohup env PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${py}" -m shotspace.cli serve --host "${host}" --port "${port}" </dev/null >> "${LOG_FILE}" 2>&1 &
  local new_pid=$!
  printf '%s\n' "${new_pid}" > "${PID_FILE}"

  # Poll local health endpoint (allow cold torch arch probe on first request)
  local health_url="http://127.0.0.1:${port}/api/health"
  for _ in {1..60}; do
    if ! kill -0 "${new_pid}" 2>/dev/null; then
      echo "[ERROR] ${APP_NAME} 启动异常退出，最近日志 (${LOG_FILE})：" >&2
      tail -n 30 "${LOG_FILE}" >&2 || true
      rm -f "${PID_FILE}"
      return 1
    fi
    if "${py}" -c "import sys, urllib.request; r = urllib.request.urlopen(sys.argv[1], timeout=8.0); sys.exit(0 if r.status == 200 else 1)" "${health_url}" 2>/dev/null; then
      local lan_ip
      lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
      [[ -z "${lan_ip}" ]] && lan_ip="127.0.0.1"
      echo "=============================================================================="
      echo "  ✓ 入镜 · ${APP_NAME} v${ver} 已成功启动 (PID ${new_pid})"
      echo "  工作台地址: http://${lan_ip}:${port}/"
      echo "  后台管理页: http://${lan_ip}:${port}/admin"
      echo "  运行日志:   ${LOG_FILE}"
      echo "=============================================================================="
      return 0
    fi
    sleep 0.5
  done

  echo "[OK] ${APP_NAME} v${ver} 已在后台启动 (PID ${new_pid})，请查看 ${LOG_FILE} 确认状态。"
}

cmd_status() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if is_running_pid "${pid}"; then
      echo "[STATUS] ${APP_NAME} v$(get_version) RUNNING (PID ${pid}) at ${APP_ROOT}"
      return 0
    fi
  fi
  echo "[STATUS] ${APP_NAME} v$(get_version) STOPPED at ${APP_ROOT}"
  return 1
}

# ------------------------------------------------------------------------------
# Pure-Code Flat Hot-Patch Packager (No .venv / weights / workspaces / secrets)
# ------------------------------------------------------------------------------
collect_code_paths() {
  local items=(
    "shotspace"
    "tests"
    "scripts"
    "docs"
    "shotspace.sh"
    "shotspace.bat"
    "pyproject.toml"
    "requirements.txt"
    "README.md"
    "config/default.json"
    "update/README.txt"
  )
  [[ -d "${APP_ROOT}/tools" ]] && items+=("tools")
  [[ -d "${APP_ROOT}/webui/dist" ]] && items+=("webui/dist")
  [[ -d "${APP_ROOT}/webui/src" ]] && items+=("webui/src")
  for f in "webui/index.html" "webui/package.json" "webui/tsconfig.json" "webui/vite.config.ts"; do
    [[ -f "${APP_ROOT}/${f}" ]] && items+=("${f}")
  done
  for p in "${items[@]}"; do
    [[ -e "${APP_ROOT}/${p}" ]] && printf '%s\n' "${p}"
  done
}

cmd_pack_patch() {
  mkdir -p "${UPDATE_DIR}"
  local ver date_tag default_name out_arg out_path
  ver="$(get_version)"
  date_tag="$(date '+%Y%m%d')"
  default_name="${APP_NAME}_${date_tag}_v${ver}_patch.tar.gz"
  out_arg="${1:-${default_name}}"

  if [[ "${out_arg}" == /* ]]; then
    out_path="$(realpath -m "${out_arg}")"
  elif [[ "${out_arg}" == */* ]]; then
    out_path="$(realpath -m "${CALLER_CWD}/${out_arg}")"
  else
    out_path="${UPDATE_DIR}/${out_arg}"
  fi
  mkdir -p "$(dirname "${out_path}")"

  mapfile -t paths < <(collect_code_paths)
  tar -czf "${out_path}" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='secrets' \
    --exclude='*.secret' \
    --exclude='*.env' \
    -C "${APP_ROOT}" "${paths[@]}"

  md5sum "${out_path}" | awk -v fn="$(basename "${out_path}")" '{print $1 "  " fn}' > "${out_path}.md5"
  local sz
  sz="$(du -h "${out_path}" | awk '{print $1}')"
  echo "[PATCH READY] ${out_path} (${sz})"
  echo "[PATCH MD5]   $(cat "${out_path}.md5")"
}

# ------------------------------------------------------------------------------
# 6-Step Safe Closed-Loop Hot Update (Strictly Enforces Fixed update/ Folder)
# ------------------------------------------------------------------------------
cmd_update() {
  mkdir -p "${UPDATE_DIR}" "${APPLIED_DIR}" "${BACKUP_DIR}"
  local patch_file=""

  if [[ $# -ge 1 && -n "${1}" ]]; then
    local cand
    if [[ -f "${1}" ]]; then
      cand="$(realpath "${1}")"
    elif [[ -f "${CALLER_CWD}/${1}" ]]; then
      cand="$(realpath "${CALLER_CWD}/${1}")"
    elif [[ -f "${UPDATE_DIR}/${1}" ]]; then
      cand="$(realpath "${UPDATE_DIR}/${1}")"
    else
      echo "[ERROR] 未找到补丁包：${1}" >&2
      return 1
    fi
    local upd_real
    upd_real="$(realpath "${UPDATE_DIR}")"
    if [[ "$(dirname "${cand}")" != "${upd_real}" ]]; then
      echo "[REJECTED] 安全限制：补丁包必须先放入固定目录 ${UPDATE_DIR}/ 下再执行 ./shotspace.sh update！" >&2
      return 1
    fi
    patch_file="${cand}"
  else
    patch_file="$(ls -1t "${UPDATE_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)"
    if [[ -z "${patch_file}" ]]; then
      echo "[ERROR] ${UPDATE_DIR}/ 目录下未找到任何待升级的 *.tar.gz 补丁包。" >&2
      return 1
    fi
  fi

  local ts
  ts="$(date '+%Y%m%d_%H%M%S')"
  echo "==> [1/6] 停服前校验补丁包完整性: $(basename "${patch_file}")"
  if ! tar -tzf "${patch_file}" >/dev/null; then
    echo "[ABORT] 补丁包损坏或格式非法，已中止升级（线上服务保持运行未受影响）！" >&2
    return 1
  fi

  # 不允许用旧版本的补丁覆盖新版本（update/ 里同时放了两个补丁、或下载错文件时会发生）；回退请用 rollback。
  local patch_ver cur_ver newest
  patch_ver="$(tar -xzOf "${patch_file}" shotspace/__init__.py 2>/dev/null | sed -n 's/^__version__ *= *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
  cur_ver="$(get_version)"
  if [[ -n "${patch_ver}" && "${patch_ver}" != "${cur_ver}" && "${SHOTSPACE_ALLOW_DOWNGRADE:-0}" != "1" ]]; then
    newest="$(printf '%s\n%s\n' "${patch_ver}" "${cur_ver}" | sort -V | tail -n 1)"
    if [[ "${newest}" == "${cur_ver}" ]]; then
      echo "[ABORT] 补丁版本 v${patch_ver} 比当前已安装的 v${cur_ver} 旧，已中止（线上服务未受影响）。" >&2
      echo "        如需回到上一版本请执行 ./shotspace.sh rollback。" >&2
      return 1
    fi
  fi

  local backup_tar="${BACKUP_DIR}/app_backup_${ts}.tar.gz"
  echo "==> [2/6] 备份当前线上代码层 -> ${backup_tar}"
  mapfile -t cur_paths < <(collect_code_paths)
  tar -czf "${backup_tar}" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -C "${APP_ROOT}" "${cur_paths[@]}"

  local was_running=0
  if cmd_status >/dev/null 2>&1; then
    was_running=1
  fi
  echo "==> [3/6] 精准停止当前目录服务实例 (${APP_ROOT})"
  cmd_stop

  echo "==> [4/6] 平铺覆盖代码层（保留 .venv / third_party / weights / workspaces / storage / run / 本机配置）"
  tar -xzf "${patch_file}" -C "${APP_ROOT}"
  chmod +x "${APP_ROOT}/shotspace.sh" 2>/dev/null || true

  echo "==> [5/6] 使用新版本脚本重定位环境并执行 Python 核心模块导入预检"
  SHOTSPACE_UPDATE_NOTES_LATER=1 "${APP_ROOT}/shotspace.sh" relocate-venv
  if ! run_import_precheck; then
    echo "[ERROR] 新版本模块导入预检失败！可执行 ./shotspace.sh rollback 一键回退至 ${backup_tar}" >&2
    return 1
  fi

  local base_stem
  base_stem="$(basename "${patch_file}" .tar.gz)"
  local archived="${APPLIED_DIR}/${base_stem}_applied_${ts}.tar.gz"
  mv -f "${patch_file}" "${archived}"
  [[ -f "${patch_file}.md5" ]] && mv -f "${patch_file}.md5" "${archived}.md5"

  echo "==> [6/6] 重启服务并归档补丁 -> ${archived}"
  if (( was_running == 1 )); then
    "${APP_ROOT}/shotspace.sh" start
  else
    echo "[OK] 代码热更新完成（原服务处于停止状态，可执行 ./shotspace.sh start 启动）。"
  fi
  # what this update changed (the new version's script prints it; several versions when the update jumped some)
  "${APP_ROOT}/shotspace.sh" release-notes --once || true
}

# ------------------------------------------------------------------------------
# Release notes: what each version changed (docs/release_notes/vX.Y.Z.txt)
#   release-notes           the installed version
#   release-notes --all     every version on record
#   release-notes X.Y.Z     that version
#   release-notes --once    every version newer than the last one shown, then remember it.  Run by `update`:
#                           an update that jumps several versions prints each of them, an update prints once.
# ------------------------------------------------------------------------------
NOTES_DIR="${APP_ROOT}/docs/release_notes"
NOTES_STATE="${APPLIED_DIR}/.notes_shown_version"

print_notes_of() {
  local v="$1" f="${NOTES_DIR}/v${1}.txt"
  [[ -f "${f}" ]] || return 1
  echo
  echo "=============================================================================="
  echo "  ${APP_NAME} v${v} 更新内容"
  echo "=============================================================================="
  cat "${f}"
  echo "=============================================================================="
  return 0
}

noted_versions() {
  local f
  for f in "${NOTES_DIR}"/v*.txt; do
    [[ -f "${f}" ]] || continue
    f="${f##*/v}"
    printf '%s\n' "${f%.txt}"
  done | sort -V
}

version_before_this_update() {
  # the version the newest backup holds: what was installed before the update that is running or just ran
  local newest
  newest="$(ls -1t "${BACKUP_DIR}"/app_backup_*.tar.gz 2>/dev/null | head -n 1 || true)"
  [[ -n "${newest}" ]] || return 0
  tar -xzOf "${newest}" shotspace/__init__.py 2>/dev/null | sed -n 's/^__version__ *= *"\([^"]*\)".*/\1/p' | head -n 1 || true
}

cmd_release_notes() {
  local mode="${1:-}" cur v last shown=0
  cur="$(get_version)"
  case "${mode}" in
    --all)
      while read -r v; do [[ -n "${v}" ]] && print_notes_of "${v}"; done < <(noted_versions)
      ;;
    --once)
      mkdir -p "${APPLIED_DIR}"
      last="$(cat "${NOTES_STATE}" 2>/dev/null | head -n 1 || true)"
      [[ -n "${last}" ]] || last="$(version_before_this_update)"
      while read -r v; do
        [[ -n "${v}" ]] || continue
        # newer than the last one shown (when that is known), and not newer than what is installed
        if [[ "$(printf '%s\n%s\n' "${v}" "${cur}" | sort -V | tail -n 1)" != "${cur}" ]]; then continue; fi
        if [[ -n "${last}" && "$(printf '%s\n%s\n' "${v}" "${last}" | sort -V | tail -n 1)" == "${last}" ]]; then continue; fi
        if [[ -z "${last}" && "${v}" != "${cur}" ]]; then continue; fi
        print_notes_of "${v}" && shown=1
      done < <(noted_versions)
      printf '%s\n' "${cur}" > "${NOTES_STATE}"
      ;;
    "")
      print_notes_of "${cur}" || echo "[INFO] 没有找到 v${cur} 的更新说明（${NOTES_DIR}）。"
      ;;
    *)
      print_notes_of "${mode#v}" || echo "[INFO] 没有找到 v${mode#v} 的更新说明（${NOTES_DIR}）。"
      ;;
  esac
}

# ------------------------------------------------------------------------------
# One-Click Rollback from Newest Backup
# ------------------------------------------------------------------------------
cmd_rollback() {
  local latest_bkp
  latest_bkp="$(ls -1t "${BACKUP_DIR}"/app_backup_*.tar.gz 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_bkp}" ]]; then
    echo "[ERROR] ${BACKUP_DIR}/ 中没有可用的历史备份包。" >&2
    return 1
  fi
  echo "==> [ROLLBACK] 校验备份包: ${latest_bkp}"
  tar -tzf "${latest_bkp}" >/dev/null

  local was_running=0
  if cmd_status >/dev/null 2>&1; then
    was_running=1
  fi
  cmd_stop
  echo "==> [ROLLBACK] 还原代码层..."
  tar -xzf "${latest_bkp}" -C "${APP_ROOT}"
  chmod +x "${APP_ROOT}/shotspace.sh" 2>/dev/null || true
  "${APP_ROOT}/shotspace.sh" relocate-venv
  run_import_precheck
  if (( was_running == 1 )); then
    "${APP_ROOT}/shotspace.sh" start
  fi
  echo "[OK] 已成功回滚至备份: $(basename "${latest_bkp}")"
}

usage() {
  cat <<EOF
Usage: ./shotspace.sh <command> [options]

Commands:
  start [--host 0.0.0.0] [--port 8765]  Relocate venv, rotate logs & start service
  stop                                  Stop service strictly under ${APP_ROOT}
  restart [--host ...] [--port ...]     Restart service
  status                                Check service status
  doctor                                Run environment & GPU health check
  relocate-venv                         Relocate .venv paths to current directory
  pack-patch [name.tar.gz]              Build lightweight flat code patch into update/
  update [update/patch.tar.gz]          6-step safe hot update from update/ directory (prints what the update changed)
  release-notes [--all | X.Y.Z]         Print what a version changed (default: the installed version)
  rollback                              One-click rollback to latest backup in backups/
EOF
}

main() {
  cd "${APP_ROOT}"
  local sub="${1:-start}"
  [[ $# -gt 0 ]] && shift
  case "${sub}" in
    start) cmd_start "$@" ;;
    stop) cmd_stop ;;
    restart) cmd_stop; cmd_start "$@" ;;
    status) cmd_status ;;
    passwd) relocate_venv; PYTHONPATH="${APP_ROOT}" "$(python_bin)" -m shotspace.cli passwd "$@" ;;
    doctor) relocate_venv; PYTHONPATH="${APP_ROOT}" "$(python_bin)" -m shotspace.cli doctor ;;
    relocate-venv)
      # Compiled bytecode left by the previous version goes first.  Python keeps a .pyc when the source has the
      # same size and the same modification second - which an updated file can have - and would then run old code
      # under the new version's name (seen once: a file that differed in one character).
      find "${APP_ROOT}/shotspace" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
      relocate_venv; echo "[OK] Relocated venv under ${APP_ROOT}"
      # An update driven by a version that does not print release notes itself (v0.2.5 and older) calls exactly
      # this command of the NEW script after unpacking: print them here.  A newer driver prints them at the end.
      if [[ "${SHOTSPACE_UPDATE_NOTES_LATER:-0}" != "1" ]]; then cmd_release_notes --once || true; fi
      ;;
    release-notes) cmd_release_notes "$@" ;;
    pack-patch) cmd_pack_patch "$@" ;;
    update) cmd_update "$@" ;;
    rollback) cmd_rollback ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
exit $?
}
