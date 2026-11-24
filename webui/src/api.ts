import {
  AdminMachine,
  AdminOverview,
  AdminTaskOwner,
  BinSlice,
  DecodedCamera,
  DecodedCharacter,
  DecodedObject,
  DecodedScene,
  DecodedSceneScan,
  MachineInfo,
  ObjectPromptItem,
  OcioOptionSet,
  OcioSettings,
  PublicQueueSummary,
  QueueJobInfo,
  ShotMeta,
  ShotSummary,
  ViewManifest,
} from "./types";

const ADMIN_TOKEN_KEY = "shotspace_admin_token";

export const ALLOWED_IMAGE_EXTS = new Set([
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
]);

export const ARCHIVE_SEQ_EXTS = new Set([".zip", ".tar", ".tgz"]);
export const VIDEO_FILE_EXTS = new Set([".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v", ".wmv"]);
export const ALLOWED_3D_EXTS = new Set([".abc", ".fbx", ".usd", ".usda", ".usdc", ".usdz"]);

export const ALLOWED_SEQ_EXTS = new Set([...ALLOWED_IMAGE_EXTS, ...ARCHIVE_SEQ_EXTS]);

export function cleanClientFilename(name: string): string {
  const s = (name || "").replace(/\\/g, "/").trim();
  const leaf = s.split("/").pop() || "";
  return leaf.trim();
}

export function isSupportedSequenceFile(name: string): boolean {
  const leaf = cleanClientFilename(name);
  const lower = leaf.toLowerCase();
  if (!lower || lower.startsWith(".") || lower === "thumbs.db" || lower === "desktop.ini") return false;
  if (lower.endsWith(".tar.gz")) return true;
  const dot = lower.lastIndexOf(".");
  if (dot < 0) return false;
  return ALLOWED_SEQ_EXTS.has(lower.slice(dot));
}

export function isSupported3dAssetFile(name: string): boolean {
  const leaf = cleanClientFilename(name).toLowerCase();
  const dot = leaf.lastIndexOf(".");
  return dot >= 0 && ALLOWED_3D_EXTS.has(leaf.slice(dot));
}

function formatFrameRangesJs(frames: number[]): string {
  if (frames.length === 0) return "";
  const vals = Array.from(new Set(frames)).sort((a, b) => a - b);
  const parts: string[] = [];
  let s = vals[0];
  let e = vals[0];
  for (let i = 1; i < vals.length; i++) {
    const v = vals[i];
    if (v === e + 1) {
      e = v;
    } else {
      parts.push(s === e ? `${s}` : `${s}–${e}`);
      s = e = v;
    }
  }
  parts.push(s === e ? `${s}` : `${s}–${e}`);
  if (parts.length > 8) return `${parts.slice(0, 8).join(", ")} 等共 ${vals.length} 帧`;
  return parts.join(", ");
}

function splitSequenceFrameTokenJs(filename: string): { stem: string; frame: number } | null {
  let m = filename.match(/^(.*?[._-])(\d{1,7})(\.[^.]+)$/);
  if (!m) {
    m = filename.match(/^(\D*)(\d{1,7})(\.[^.]+)$/);
  }
  if (!m) return null;
  const stem = m[1].replace(/[._-]+$/, "").toLowerCase();
  return { stem, frame: parseInt(m[2], 10) };
}

/**
 * Fast pre-upload validation & interception (`< 5 ms` before transferring gigabytes over LAN):
 * catches video files, 0-byte broken files, `< 2` single-image mistakes, unnumbered stills,
 * mixed sequence prefixes, duplicate frame numbers, and missing frames (`缺帧`).
 */
export function validateAndFilterSequenceFiles(rawFiles: File[]): File[] {
  const nonSystem = rawFiles.filter((f) => {
    const leaf = cleanClientFilename(f.name).toLowerCase();
    return leaf && !leaf.startsWith(".") && leaf !== "thumbs.db" && leaf !== "desktop.ini";
  });

  const valid = nonSystem.filter((f) => isSupportedSequenceFile(f.name));
  if (valid.length === 0) {
    const vid = nonSystem.find((f) => {
      const leaf = cleanClientFilename(f.name).toLowerCase();
      const dot = leaf.lastIndexOf(".");
      return dot >= 0 && VIDEO_FILE_EXTS.has(leaf.slice(dot));
    });
    if (vid) {
      throw new Error(
        `素材校验失败：暂不支持直接导入视频文件（检测到 ${cleanClientFilename(vid.name)}），请拖入去畸变序列帧文件夹或 .zip/.tar 序列包。`
      );
    }
    throw new Error(
      "素材校验失败：未发现支持的序列帧文件（支持 .exr, .dpx, .cin, .tif, .png, .jpg, .tga, .bmp, .webp, .hdr 或 .zip/.tar 序列包）。"
    );
  }

  const zeroFiles = valid.filter((f) => f.size === 0);
  if (zeroFiles.length > 0) {
    const sample = zeroFiles
      .slice(0, 3)
      .map((f) => cleanClientFilename(f.name))
      .join(", ");
    throw new Error(
      `素材校验失败：发现空文件 / 损坏帧（0 字节）：${sample}${zeroFiles.length > 3 ? ` 等共 ${zeroFiles.length} 个文件` : ""}。`
    );
  }

  const archives = valid.filter((f) => {
    const lower = cleanClientFilename(f.name).toLowerCase();
    const dot = lower.lastIndexOf(".");
    return lower.endsWith(".tar.gz") || (dot >= 0 && ARCHIVE_SEQ_EXTS.has(lower.slice(dot)));
  });
  if (archives.length > 0) {
    // Archive contents (missing frames, resolution consistency, corruption) are validated on extraction
    return archives;
  }

  // Group image files by extension and validate the dominant sequence before uploading
  const extCounts = new Map<string, File[]>();
  for (const f of valid) {
    const leaf = cleanClientFilename(f.name).toLowerCase();
    const ext = leaf.slice(leaf.lastIndexOf("."));
    const arr = extCounts.get(ext) || [];
    arr.push(f);
    extCounts.set(ext, arr);
  }
  let domExt = "";
  let domFiles: File[] = [];
  for (const [ext, list] of extCounts.entries()) {
    if (list.length > domFiles.length) {
      domExt = ext;
      domFiles = list;
    }
  }
  const otherExts = Array.from(extCounts.entries()).filter(([k]) => k !== domExt);

  if (domFiles.length < 2) {
    const onlyName = domFiles[0] ? cleanClientFilename(domFiles[0].name) : "";
    const extraHint =
      otherExts.length > 0
        ? `（另检测到其他格式 ${otherExts.map(([k, v]) => `${k}×${v.length}`).join(", ")}，序列后缀须统一）`
        : "";
    throw new Error(
      `素材校验失败：序列解算至少需要连续 2 帧以上，当前仅选择了 ${domFiles.length} 帧图片${onlyName ? `（${onlyName}）` : ""}${extraHint}；请拖入整个序列文件夹或 Ctrl+A 全选序列帧。`
    );
  }

  const unnumbered: string[] = [];
  const stems = new Map<string, number>();
  const frameMap = new Map<number, string[]>();

  for (const f of domFiles) {
    const leaf = cleanClientFilename(f.name);
    const tok = splitSequenceFrameTokenJs(leaf);
    if (!tok) {
      unnumbered.push(leaf);
    } else {
      stems.set(tok.stem, (stems.get(tok.stem) || 0) + 1);
      const list = frameMap.get(tok.frame) || [];
      list.push(leaf);
      frameMap.set(tok.frame, list);
    }
  }

  if (unnumbered.length > 0) {
    throw new Error(
      `素材校验失败：序列中混入无帧号命名的图片（${unnumbered.slice(0, 4).join(", ")}${unnumbered.length > 4 ? ` 等 ${unnumbered.length} 个` : ""}），请清理非序列单张图后重新上传。`
    );
  }

  if (stems.size > 1) {
    const desc = Array.from(stems.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([st, cnt]) => `${st || "<纯数字>"}* (${cnt}帧)`)
      .join("、");
    throw new Error(`素材校验失败：检测到多个不同命名的序列混杂（${desc}），请每次仅上传同一镜头的序列帧。`);
  }

  const dupFrames = Array.from(frameMap.entries())
    .filter(([, names]) => names.length > 1)
    .map(([f]) => f)
    .sort((a, b) => a - b);
  if (dupFrames.length > 0) {
    const f0 = dupFrames[0];
    const dupNames = (frameMap.get(f0) || []).slice(0, 3).join(", ");
    throw new Error(
      `素材校验失败：检测到重复帧号（共 ${dupFrames.length} 处：${formatFrameRangesJs(dupFrames)}，如帧 ${f0} 存在多个文件：${dupNames}），请清理重复文件后重新上传。`
    );
  }

  const sortedFrames = Array.from(frameMap.keys()).sort((a, b) => a - b);
  const minF = sortedFrames[0];
  const maxF = sortedFrames[sortedFrames.length - 1];
  const expectedTotal = maxF - minF + 1;
  if (expectedTotal !== sortedFrames.length) {
    const present = new Set(sortedFrames);
    const missing: number[] = [];
    for (let f = minF; f <= maxF; f++) {
      if (!present.has(f)) missing.push(f);
    }
    const otherNote =
      otherExts.length > 0
        ? `（同时发现其他后缀文件 ${otherExts.map(([k, v]) => `${k}×${v.length}`).join(", ")}，请检查是否有帧渲染成了不同格式）`
        : "";
    throw new Error(
      `素材校验失败：序列存在缺帧：帧范围 ${minF}–${maxF} 应有 ${expectedTotal} 帧，实选 ${sortedFrames.length} 帧，缺失 ${missing.length} 帧：${formatFrameRangesJs(missing)}${otherNote}。`
    );
  }

  // Return all valid image files (including any minor extra image files so server records advisory warning)
  return valid;
}

export function getAdminToken(): string {
  return localStorage.getItem(ADMIN_TOKEN_KEY) || "";
}

export function setAdminToken(token: string): void {
  if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token);
  else localStorage.removeItem(ADMIN_TOKEN_KEY);
}

async function jsonReq<T>(url: string, init?: RequestInit, useAdminAuth = false): Promise<T> {
  const headers = new Headers(init?.headers || {});
  if (useAdminAuth) {
    const tok = getAdminToken();
    if (tok) headers.set("Authorization", `Bearer ${tok}`);
  }
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    const err: any = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json() as Promise<T>;
}

function xhrUploadJson<T>(
  url: string,
  file: File,
  onProgress?: (pct: number, phase: "uploading" | "parsing") => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable && ev.total > 0) {
        const pct = Math.min(100, Math.round((ev.loaded / ev.total) * 100));
        onProgress?.(pct, pct >= 100 ? "parsing" : "uploading");
      }
    };
    xhr.upload.onload = () => {
      onProgress?.(100, "parsing");
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch (err) {
          reject(err);
        }
      } else {
        let detail = `HTTP ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          detail = body.detail || JSON.stringify(body);
        } catch {
          detail = xhr.responseText || detail;
        }
        const err: any = new Error(detail);
        err.status = xhr.status;
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("网络传输中断，上传失败"));
    onProgress?.(0, "uploading");
    xhr.send(file);
  });
}

export const api = {
  health: () =>
    jsonReq<{
      status: string;
      version: string;
      initialized: boolean;
      configured_port: number;
      torch_version: string;
      cuda_archs: string[];
      ocio: OcioOptionSet;
      queue: PublicQueueSummary;
      toolchains: Record<string, boolean>;
    }>("/api/health"),

  listShots: () => jsonReq<{ shots: ShotSummary[]; queue: PublicQueueSummary; machine?: MachineInfo }>("/api/shots"),

  /** v0.2.12: a public task goes into this computer's own list */
  claimShot: (id: string) => jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/claim`, { method: "POST" }),

  getShot: (id: string) => jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}`),

  createShot: (name: string) =>
    jsonReq<ShotMeta>("/api/shots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  updateShot: (id: string, patch: Partial<ShotMeta>) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteShot: (id: string) =>
    jsonReq<{ ok: boolean }>(`/api/shots/${encodeURIComponent(id)}`, { method: "DELETE" }),

  uploadSequenceFiles: async (
    id: string,
    rawFiles: File[],
    ocio: OcioSettings,
    fps = 24,
    onProgress?: (done: number, total: number) => void
  ): Promise<ShotMeta> => {
    const valid = validateAndFilterSequenceFiles(rawFiles);
    await jsonReq<{ ok: boolean }>(`/api/shots/${encodeURIComponent(id)}/sequence/begin`, {
      method: "POST",
    });

    let done = 0;
    const total = valid.length;
    onProgress?.(0, total);
    let cursor = 0;
    const concurrency = Math.min(6, total);

    const worker = async () => {
      while (cursor < total) {
        const idx = cursor++;
        const file = valid[idx];
        const qs = new URLSearchParams({ filename: cleanClientFilename(file.name) });
        await jsonReq(`/api/shots/${encodeURIComponent(id)}/sequence/frame?${qs}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file,
        });
        done += 1;
        onProgress?.(done, total);
      }
    };

    await Promise.all(Array.from({ length: concurrency }, () => worker()));

    return jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/sequence/finalize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input_colorspace: ocio.input_colorspace,
        ocio_view: ocio.ocio_view,
        fps,
      }),
    });
  },

  updateOcio: (id: string, ocio: OcioSettings) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/ocio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ocio),
    }),

  uploadCamera: (
    id: string,
    file: File,
    opts: {
      unit?: string;
      camera_path?: string;
      fbx_take?: string;
      frame_offset?: number;
      onProgress?: (pct: number, phase: "uploading" | "parsing") => void;
    } = {}
  ) => {
    const qs = new URLSearchParams({
      filename: cleanClientFilename(file.name) || "camera.abc",
      unit: opts.unit || "m",
      camera_path: opts.camera_path || "",
      fbx_take: opts.fbx_take || "",
      frame_offset: String(opts.frame_offset || 0),
    });
    return xhrUploadJson<ShotMeta>(
      `/api/shots/${encodeURIComponent(id)}/camera/upload?${qs}`,
      file,
      opts.onProgress
    );
  },

  updateCameraOptions: (
    id: string,
    opts: { camera_path?: string; unit?: string; fbx_take?: string; frame_offset?: number }
  ) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/camera/options`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_path: opts.camera_path || "",
        unit: opts.unit || "",
        fbx_take: opts.fbx_take || "",
        frame_offset: opts.frame_offset || 0,
      }),
    }),

  clearCamera: (id: string) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/camera`, { method: "DELETE" }),

  uploadSceneScan: (
    id: string,
    file: File,
    opts: {
      unit?: string;
      onProgress?: (pct: number, phase: "uploading" | "parsing") => void;
    } = {}
  ) => {
    const qs = new URLSearchParams({
      filename: cleanClientFilename(file.name) || "scene_scan.abc",
      unit: opts.unit || "m",
    });
    return xhrUploadJson<ShotMeta>(
      `/api/shots/${encodeURIComponent(id)}/scene_scan/upload?${qs}`,
      file,
      opts.onProgress
    );
  },

  updateSceneScanOptions: (
    id: string,
    opts: { unit?: string; selected_paths?: string[] }
  ) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/scene_scan/options`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        unit: opts.unit || "",
        selected_paths: opts.selected_paths,
      }),
    }),

  clearSceneScan: (id: string) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/scene_scan`, { method: "DELETE" }),

  /** Step 4: put the solved people on the scan (true), or show the task as it is without it (false). */
  setScanAlignment: (id: string, enabled: boolean) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/scene_scan/align`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }),

  removeSolvedPeople: (id: string, personIds: number[] | null) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/solved_people/remove`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ person_ids: personIds }),
    }),

  runDetect: (id: string, threshold: number, grade: ShotMeta["grade"]) =>
    jsonReq<{ queued: boolean; job: QueueJobInfo; shot: ShotMeta }>(
      `/api/shots/${encodeURIComponent(id)}/detect`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold, grade }),
      }
    ),

  runSolve: (
    id: string,
    payload: {
      grade: ShotMeta["grade"];
      detect_params: ShotMeta["detect_params"];
      manual_people?: ShotMeta["manual_people"];
      selection: ShotMeta["selection"];
      solve_params: ShotMeta["solve_params"];
    }
  ) =>
    jsonReq<{ queued: boolean; job: QueueJobInfo; shot: ShotMeta }>(
      `/api/shots/${encodeURIComponent(id)}/solve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),

  runObjectTrack: (
    id: string,
    object_prompts: ObjectPromptItem[],
    grade: ShotMeta["grade"]
  ) =>
    jsonReq<{ queued: boolean; job: QueueJobInfo; shot: ShotMeta }>(
      `/api/shots/${encodeURIComponent(id)}/objects/track`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object_prompts, grade }),
      }
    ),

  runObjectSolve: (
    id: string,
    payload: {
      object_prompts: ObjectPromptItem[];
      object_solve_params: { smoothing: number };
      grade: ShotMeta["grade"];
    }
  ) =>
    jsonReq<{ queued: boolean; job: QueueJobInfo; shot: ShotMeta }>(
      `/api/shots/${encodeURIComponent(id)}/objects/solve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),

  cancelShot: (id: string) =>
    jsonReq<ShotMeta>(`/api/shots/${encodeURIComponent(id)}/cancel`, { method: "POST" }),

  downloadSingleFormat: async (id: string, fmt: string, unit = ""): Promise<ShotMeta> => {
    const res = await jsonReq<{ exports: ShotMeta["exports"]; shot: ShotMeta }>(
      `/api/shots/${encodeURIComponent(id)}/export`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ formats: [fmt], unit }),
      }
    );
    const item = res.exports[0];
    if (item) {
      const a = document.createElement("a");
      a.href = `/api/shots/${encodeURIComponent(id)}/exports/${encodeURIComponent(item.filename)}`;
      a.download = item.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
    return res.shot;
  },

  // ------------------------------------------------------------------ `/api/admin/*`

  adminStatus: () =>
    jsonReq<{
      initialized: boolean;
      authenticated: boolean;
      port: number;
      host: string;
      default_cache_dir?: string;
    }>("/api/admin/status", undefined, true),

  adminSetup: (port: number, password: string, cacheDir = "") =>
    jsonReq<{ ok: boolean; port: number; cache_dir?: string; token: string }>("/api/admin/setup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ port, password, cache_dir: cacheDir }),
    }),

  adminLogin: (password: string) =>
    jsonReq<{ ok: boolean; token: string; port: number }>("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }),

  adminOverview: () => jsonReq<AdminOverview>("/api/admin/overview", undefined, true),

  /** v0.2.12: every task with the computer it belongs to */
  adminTaskOwners: () => jsonReq<{ shots: AdminTaskOwner[] }>("/api/admin/shots", undefined, true),

  adminMachines: () => jsonReq<{ machines: AdminMachine[] }>("/api/admin/machines", undefined, true),

  adminMakeTaskUnclassified: (id: string) =>
    jsonReq<{ id: string; owner: string }>(`/api/admin/shots/${encodeURIComponent(id)}/unclassified`, { method: "POST" }, true),

  adminAssignTask: (id: string, ip: string, name: string) =>
    jsonReq<{ id: string; owner: string }>(
      `/api/admin/shots/${encodeURIComponent(id)}/owner`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ip, name }) },
      true
    ),

  adminDeleteTask: (id: string) =>
    jsonReq<{ id: string; deleted: boolean }>(`/api/admin/shots/${encodeURIComponent(id)}`, { method: "DELETE" }, true),

  adminToggleGpu: (gpuIndex: number, enabled: boolean) =>
    jsonReq<AdminOverview>(
      `/api/admin/gpus/${gpuIndex}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      },
      true
    ),

  adminToggleAllGpus: (enabled: boolean) =>
    jsonReq<AdminOverview>(
      "/api/admin/gpus/all",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      },
      true
    ),

  adminCancelJob: (jobId: string) =>
    jsonReq<{ ok: boolean; overview: AdminOverview }>(
      `/api/admin/jobs/${encodeURIComponent(jobId)}/cancel`,
      { method: "POST" },
      true
    ),

  adminChangePassword: (oldPassword: string, newPassword: string) =>
    jsonReq<{ ok: boolean; token: string }>(
      "/api/admin/password",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
      },
      true
    ),

  adminSetPort: (port: number) =>
    jsonReq<{ ok: boolean; port: number }>(
      "/api/admin/port",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port }),
      },
      true
    ),

  adminSetCacheDir: (cacheDir: string) =>
    jsonReq<{
      ok: boolean;
      storage: NonNullable<AdminOverview["server"]>;
    }>(
      "/api/admin/cache_dir",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cache_dir: cacheDir }),
      },
      true
    ),

  adminSetRetention: (retentionDays: number, runCleanup = true) =>
    jsonReq<{
      ok: boolean;
      retention_days: number;
      cleaned: { deleted_count: number; deleted_ids: string[] };
      storage: { retention_days: number; total_mb: number; shots_total: number; solved_shots: number };
    }>(
      "/api/admin/retention",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ retention_days: retentionDays, run_cleanup: runCleanup }),
      },
      true
    ),

  loadSceneBundle: async (id: string): Promise<DecodedScene> => {
    const [manifest, buf] = await Promise.all([
      jsonReq<ViewManifest>(`/api/shots/${encodeURIComponent(id)}/view/manifest?t=${Date.now()}`),
      fetch(`/api/shots/${encodeURIComponent(id)}/view/scene.bin?t=${Date.now()}`).then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch scene.bin (${r.status})`);
        return r.arrayBuffer();
      }),
    ]);

    const f32 = (s: BinSlice) => new Float32Array(buf, s.offset, s.length);
    const u32 = (s: BinSlice) => new Uint32Array(buf, s.offset, s.length);
    const u16 = (s: BinSlice) => new Uint16Array(buf, s.offset, s.length);

    const cameras: DecodedCamera[] = manifest.cameras.map((c) => {
      const frameToIndex = new Map<number, number>();
      c.frames.forEach((f, idx) => frameToIndex.set(f, idx));
      return {
        role: c.role,
        name: c.name,
        path: c.path,
        frames: c.frames,
        frameToIndex,
        width: c.width,
        height: c.height,
        pixel_aspect: c.pixel_aspect,
        overscan: c.overscan,
        fps: c.fps,
        world: f32(c.world),
        focal_mm: f32(c.focal_mm),
        h_aperture_mm: f32(c.h_aperture_mm),
        v_aperture_mm: f32(c.v_aperture_mm),
        center_mm: f32(c.center_mm),
      };
    });

    const characters: DecodedCharacter[] = manifest.characters.map((ch) => {
      const frameToIndex = new Map<number, number>();
      ch.frames.forEach((f, idx) => frameToIndex.set(f, idx));
      return {
        name: ch.name,
        person_id: ch.person_id,
        frames: ch.frames,
        visible: ch.visible || ch.frames.map(() => 1),
        frameToIndex,
        joint_names: ch.joint_names,
        parents: ch.parents,
        rest_points: f32(ch.rest_points),
        faces: u32(ch.faces),
        skin_indices: u16(ch.skin_indices),
        skin_weights: f32(ch.skin_weights),
        bind_world: f32(ch.bind_world),
        inv_bind: f32(ch.inv_bind),
        anim_world: f32(ch.anim_world),
      };
    });

    const objects: DecodedObject[] = (manifest.objects || []).map((o) => {
      const frameToIndex = new Map<number, number>();
      o.frames.forEach((f, idx) => frameToIndex.set(f, idx));
      return {
        name: o.name,
        object_id: o.object_id,
        frames: o.frames,
        visible: o.visible || o.frames.map(() => 1),
        frameToIndex,
        rest_points: f32(o.rest_points),
        faces: u32(o.faces),
        vertex_colors: o.vertex_colors ? f32(o.vertex_colors) : null,
        world: f32(o.world),
      };
    });

    const scene_scan: DecodedSceneScan | null = manifest.scene_scan
      ? {
          name: manifest.scene_scan.name,
          vertex_count: manifest.scene_scan.vertex_count,
          face_count: manifest.scene_scan.face_count,
          vertices: f32(manifest.scene_scan.vertices),
          faces: u32(manifest.scene_scan.faces),
        }
      : null;

    return { manifest, cameras, characters, objects, scene_scan };
  },
};
