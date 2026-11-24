import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  cleanClientFilename,
  isSupported3dAssetFile,
  validateAndFilterSequenceFiles,
} from "./api";
import { AdminPage } from "./components/AdminPage";
import { ConfirmHost, confirmAction } from "./components/Confirm";
import { CameraHierarchyModal, SceneScanHierarchyModal } from "./components/HierarchyModals";
import { Range } from "./components/Range";
import {
  CoffeeCup,
  IconCamera,
  IconCheck,
  IconChevronDown,
  IconChevronLeft,
  IconChevronRight,
  IconClose,
  IconPause,
  IconPlay,
  IconSkipEnd,
  IconSkipStart,
} from "./components/Icons";
import { NewTaskModal, TaskGalleryModal } from "./components/TaskModals";
import {
  DecodedScene,
  DetectedPerson,
  GradeParams,
  ManualPersonTrack,
  ObjectPromptItem,
  OcioOptionSet,
  OcioSettings,
  PublicQueueSummary,
  SelectionState,
  ShotMeta,
  ShotSummary,
  TRACKER_ENGINE,
  WorkMode,
  objectColor,
  personColor,
} from "./types";
import {
  buildManualDetectedPeople,
  normalizeManualPeopleList,
  removeManualKeyframeAt,
  upsertManualKeyframe,
} from "./utils/manualBoxes";
import { CharacterDisplay, Stage3D, ViewMode } from "./viewport/Stage3D";

function computeSelectedIds(people: DetectedPerson[], sel: SelectionState): Set<number> {
  const minF = Math.max(1, sel.least_frames || 1);
  const autoPeople = people.filter((p) => !p.is_manual);
  const manualValid = people.filter((p) => Boolean(p.is_manual) && p.frame_count >= 1);

  let autoValid = autoPeople.filter((p) => p.frame_count >= minF);
  if (autoValid.length === 0 && autoPeople.length > 0) {
    autoValid = autoPeople.filter((p) => p.frame_count >= 1);
  }
  const valid = [...autoValid, ...manualValid];

  if (sel.mode === "largest") {
    return new Set([...autoValid.slice(0, 1), ...manualValid].map((p) => p.id));
  }
  if (sel.mode === "top") {
    return new Set([...autoValid.slice(0, Math.max(1, sel.count)), ...manualValid].map((p) => p.id));
  }
  if (sel.mode === "all") return new Set(valid.map((p) => p.id));
  if (sel.mode === "ids") {
    const out = new Set<number>();
    (sel.ids || "").split(/[,;\s]+/).forEach((tok) => {
      if (!tok) return;
      if (tok.includes("-")) {
        const [a, b] = tok.split("-").map(Number);
        if (!isNaN(a) && !isNaN(b)) {
          for (let k = Math.min(a, b); k <= Math.max(a, b); k++) out.add(k);
        }
      } else {
        const n = Number(tok);
        if (!isNaN(n)) out.add(n);
      }
    });
    return new Set(valid.filter((p) => out.has(p.id)).map((p) => p.id));
  }
  if (sel.mode === "picked") {
    if ((sel.picks || []).length === 0 && manualValid.length > 0) {
      return new Set(manualValid.map((p) => p.id));
    }
    const picked = new Set<number>();
    for (const pk of sel.picks || []) {
      const [f, px, py] = pk;
      let bestId = -1;
      let bestArea = Infinity;
      for (const p of valid) {
        const b = p.boxes[String(Math.round(f))];
        if (!b) continue;
        if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]) {
          const area = (b[2] - b[0]) * (b[3] - b[1]);
          if (area < bestArea) {
            bestArea = area;
            bestId = p.id;
          }
        }
      }
      if (bestId > 0) picked.add(bestId);
    }
    return picked;
  }
  return new Set();
}

async function collectDroppedFiles(dataTransfer: DataTransfer): Promise<File[]> {
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((it) => (typeof (it as any).webkitGetAsEntry === "function" ? (it as any).webkitGetAsEntry() : null))
    .filter(Boolean);

  if (entries.length === 0) {
    return Array.from(dataTransfer.files || []);
  }

  const out: File[] = [];
  const walk = async (entry: any): Promise<void> => {
    if (entry.isFile) {
      const file: File | null = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
      if (file) out.push(file);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      while (true) {
        const batch: any[] = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
        if (!batch.length) break;
        for (const child of batch) {
          await walk(child);
        }
      }
    }
  };
  for (const e of entries) {
    await walk(e);
  }
  return out;
}

const ACTIVE_SHOT_STORAGE_KEY = "shotspace_active_shot_id";

function readRememberedShotId(): string {
  try {
    const q = new URLSearchParams(window.location.search).get("shot");
    if (q && q.trim()) return q.trim();
  } catch {}
  try {
    return localStorage.getItem(ACTIVE_SHOT_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function rememberActiveShotId(id: string): void {
  if (!id) return;
  try {
    localStorage.setItem(ACTIVE_SHOT_STORAGE_KEY, id);
  } catch {}
  if (!window.location.pathname.startsWith("/admin")) {
    const targetUrl = `/?shot=${encodeURIComponent(id)}`;
    if (`${window.location.pathname}${window.location.search}` !== targetUrl) {
      window.history.replaceState(window.history.state, "", targetUrl);
    }
  }
}

function readRememberedShotViewState(id: string): { frame?: number; viewMode?: ViewMode } | null {
  try {
    const raw = sessionStorage.getItem(`shotspace_view_${id}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function rememberShotViewState(id: string, frame: number, viewMode: ViewMode): void {
  if (!id) return;
  try {
    sessionStorage.setItem(`shotspace_view_${id}`, JSON.stringify({ frame, viewMode }));
  } catch {}
}

export const App: React.FC = () => {
  const [route, setRoute] = useState<"workbench" | "admin">(() =>
    window.location.pathname.startsWith("/admin") ? "admin" : "workbench"
  );

  useEffect(() => {
    const onPop = () => {
      setRoute(window.location.pathname.startsWith("/admin") ? "admin" : "workbench");
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigateTo = useCallback((target: "workbench" | "admin") => {
    if (target === "admin") {
      if (window.location.pathname !== "/admin") {
        window.history.pushState({}, "", "/admin");
      }
    } else {
      const savedId = readRememberedShotId();
      const targetUrl = savedId ? `/?shot=${encodeURIComponent(savedId)}` : "/";
      if (`${window.location.pathname}${window.location.search}` !== targetUrl) {
        window.history.pushState({}, "", targetUrl);
      }
    }
    setRoute(target);
  }, []);

  if (route === "admin") {
    return <AdminPage onNavigateWorkbench={() => navigateTo("workbench")} />;
  }

  return <WorkbenchView onNavigateAdmin={() => navigateTo("admin")} />;
};

const WorkbenchView: React.FC<{ onNavigateAdmin: () => void }> = ({ onNavigateAdmin }) => {
  const [ocioOptions, setOcioOptions] = useState<OcioOptionSet | null>(null);
  // the version the SERVICE runs (not the one this page was built from): shown under the name in the top bar
  const [appVersion, setAppVersion] = useState<string>("");
  const [queueSummary, setQueueSummary] = useState<PublicQueueSummary | null>(null);
  const [shots, setShots] = useState<ShotSummary[]>([]);
  // v0.2.12: the computer this page runs on, as the server sees it (tasks belong to computers)
  const [machine, setMachine] = useState<{ ip: string; name: string } | null>(null);
  const [claimingShot, setClaimingShot] = useState(false);
  const [shot, setShot] = useState<ShotMeta | null>(null);
  const [scene, setScene] = useState<DecodedScene | null>(null);

  // Modals (no ugly browser window.prompt or native top <select>!)
  const [showGalleryModal, setShowGalleryModal] = useState(false);
  const [showNewTaskModal, setShowNewTaskModal] = useState(false);
  const [showCamTreeModal, setShowCamTreeModal] = useState(false);
  const [showScanTreeModal, setShowScanTreeModal] = useState(false);
  const [step2Highlight, setStep2Highlight] = useState(false);

  // Step 2 & Step 5: Upload / Parsing progress + studio-default Meter ("m") unit
  const [uploadingCam, setUploadingCam] = useState<{
    name: string;
    pct: number;
    phase: "uploading" | "parsing";
  } | null>(null);
  const [updatingCamOpts, setUpdatingCamOpts] = useState(false);
  const [uploadingScan, setUploadingScan] = useState<{
    name: string;
    pct: number;
    phase: "uploading" | "parsing";
  } | null>(null);
  const [updatingScanOpts, setUpdatingScanOpts] = useState(false);
  const [switchingScanAlign, setSwitchingScanAlign] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [exportUnit, setExportUnit] = useState<"m" | "cm">("m");
  const [exportError, setExportError] = useState<string | null>(null);

  // Step 1: Sequence upload & OCIO & Grade
  const [uploadingSeq, setUploadingSeq] = useState<{ done: number; total: number } | null>(null);
  const [ocio, setOcio] = useState<OcioSettings>({ input_colorspace: "auto", ocio_view: "aces_sdr" });
  const [updatingOcio, setUpdatingOcio] = useState(false);
  const [grade, setGrade] = useState<GradeParams>({
    exposure: 0,
    brightness: 0,
    contrast: 1,
    saturation: 1,
    gamma: 1,
  });
  const [bypassGrade, setBypassGrade] = useState(false);

  // Mode Switcher ("human" = 人物解算, "object" = 物体匹配)
  const [workMode, setWorkMode] = useState<WorkMode>("human");

  // Step 3 & 4 (Human): Detect & Select + Manual Keyframe Boxes + SAM 3D Body
  const [threshold, setThreshold] = useState(0.8);
  const [manualPeople, setManualPeople] = useState<ManualPersonTrack[]>([]);
  const [activeManualId, setActiveManualId] = useState<number | null>(null);
  const [manualDrawMode, setManualDrawMode] = useState<boolean>(false);
  const [selection, setSelection] = useState<SelectionState>({
    mode: "largest",
    count: 2,
    ids: "1",
    picks: [],
    least_frames: 3,
  });
  const [handRefine, setHandRefine] = useState(true);
  const [useMask, setUseMask] = useState(true);
  const [lockShape, setLockShape] = useState(true);
  const [smoothing, setSmoothing] = useState(0.5);

  // Step 3 & 4 (Object): SAM 3.1 Interactive Click Tracking + 3D Proxy & 6DoF Pose
  const [objectPrompts, setObjectPrompts] = useState<ObjectPromptItem[]>([
    { id: 1, name: "object_01", points: [] },
  ]);
  const [activeObjectId, setActiveObjectId] = useState<number>(1);
  const [objectClickLabel, setObjectClickLabel] = useState<0 | 1>(1);
  const [objectSmoothing, setObjectSmoothing] = useState<number>(0.5);
  const [showObjectMasks, setShowObjectMasks] = useState<boolean>(true);

  // Step 5: export (each file is built when it is downloaded)
  const [downloadingFmt, setDownloadingFmt] = useState<string | null>(null);

  // Viewport & Timeline state
  const [viewMode, setViewMode] = useState<ViewMode>("camera");
  const [displayMode, setDisplayMode] = useState<CharacterDisplay>("both");
  const [wipeEnabled, setWipeEnabled] = useState<boolean>(false);
  const [wipeRatio, setWipeRatio] = useState<number>(0.5);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.78);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showSceneScan, setShowSceneScan] = useState<boolean>(true);
  const [autoHide, setAutoHide] = useState<boolean>(true);
  const [gateZoom, setGateZoom] = useState<number>(1.0);
  const [gatePan, setGatePan] = useState<[number, number]>([0, 0]);
  const [frame, setFrame] = useState<number>(1001);
  const [playing, setPlaying] = useState<boolean>(false);
  const [uiNotice, setUiNotice] = useState<string | null>(null);
  const [seqInterceptError, setSeqInterceptError] = useState<string | null>(null);
  const [dragHoverTarget, setDragHoverTarget] = useState<string | null>(null);

  const step2Ref = useRef<HTMLElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const filesInputRef = useRef<HTMLInputElement>(null);
  const camInputRef = useRef<HTMLInputElement>(null);
  const scanInputRef = useRef<HTMLInputElement>(null);
  const loadedShotIdRef = useRef<string | null>(null);
  const lastCamUnitSigRef = useRef<string>("");
  const pendingPatchRef = useRef<{ shotId: string; patch: Partial<ShotMeta> } | null>(null);
  const saveTimerRef = useRef<number | null>(null);

  const flushPendingPatch = useCallback(async () => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    const pending = pendingPatchRef.current;
    pendingPatchRef.current = null;
    if (!pending) return;
    try {
      await api.updateShot(pending.shotId, pending.patch);
    } catch {
      // ignore background persistence errors
    }
  }, []);

  const scheduleSavePatch = useCallback(
    (patch: Partial<ShotMeta>) => {
      if (!shot?.id) return;
      if (pendingPatchRef.current && pendingPatchRef.current.shotId !== shot.id) {
        void flushPendingPatch();
      }
      pendingPatchRef.current = {
        shotId: shot.id,
        patch: { ...(pendingPatchRef.current?.patch || {}), ...patch },
      };
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
      }
      saveTimerRef.current = window.setTimeout(() => {
        void flushPendingPatch();
      }, 250);
    },
    [shot?.id, flushPendingPatch]
  );

  useEffect(() => {
    const onBeforeUnload = () => {
      if (pendingPatchRef.current) {
        const { shotId, patch } = pendingPatchRef.current;
        pendingPatchRef.current = null;
        fetch(`/api/shots/${encodeURIComponent(shotId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
          keepalive: true,
        }).catch(() => {});
      }
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      void flushPendingPatch();
    };
  }, [flushPendingPatch]);

  const syncLocalFromShot = useCallback((m: ShotMeta, forceAll = false) => {
    const isNewShot = forceAll || loadedShotIdRef.current !== m.id;
    setShot(m);

    const camSig = m.camera
      ? `${m.camera.file_path}:${m.camera.unit || ""}:${m.camera.resolved_unit_to_cm ?? ""}`
      : "";
    if (isNewShot || camSig !== lastCamUnitSigRef.current) {
      lastCamUnitSigRef.current = camSig;
      if (m.camera) {
        const cu = m.camera.unit;
        const rCm = m.camera.resolved_unit_to_cm ?? m.camera.listing?.unit_to_cm ?? 100;
        if (cu === "m" || (cu === "auto" && Math.abs(rCm - 100) < 1)) {
          setExportUnit("m");
        } else if (cu === "cm" || (cu === "auto" && Math.abs(rCm - 1) < 0.1)) {
          setExportUnit("cm");
        }
      } else if (isNewShot) {
        setExportUnit("m");
      }
    }

    if (!isNewShot) {
      return;
    }

    loadedShotIdRef.current = m.id;
    rememberActiveShotId(m.id);
    setWorkMode(m.work_mode || "human");
    setOcio(m.ocio || { input_colorspace: "auto", ocio_view: "aces_sdr" });
    setGrade({
      exposure: m.grade?.exposure ?? 0,
      brightness: m.grade?.brightness ?? 0,
      contrast: m.grade?.contrast ?? 1,
      saturation: m.grade?.saturation ?? 1,
      gamma: m.grade?.gamma ?? 1,
    });
    setThreshold(m.detect_params?.threshold ?? 0.8);
    const mps = normalizeManualPeopleList(m.manual_people || []);
    setManualPeople(mps);
    setActiveManualId((prev) =>
      prev !== null && mps.some((p) => p.id === prev) ? prev : mps[0]?.id ?? null
    );
    if (mps.length === 0) {
      setManualDrawMode(false);
    }
    setSelection(
      m.selection || { mode: "largest", count: 2, ids: "1", picks: [], least_frames: 3 }
    );
    setHandRefine(m.solve_params?.hand_refine ?? true);
    setUseMask(true);
    setLockShape(true);
    setSmoothing(m.solve_params?.smoothing ?? 0.5);
    const ops =
      m.object_prompts && m.object_prompts.length > 0
        ? m.object_prompts
        : [{ id: 1, name: "object_01", points: [] }];
    setObjectPrompts(ops);
    setActiveObjectId((prev) => (ops.some((o) => o.id === prev) ? prev : ops[0].id));
    setObjectSmoothing(m.object_solve_params?.smoothing ?? 0.5);
  }, []);

  const loadSceneIfReady = useCallback(async (m: ShotMeta) => {
    if (!m.has_view) {
      setScene(null);
      return;
    }
    try {
      const bundle = await api.loadSceneBundle(m.id);
      setScene(bundle);
      if (bundle.manifest.frames.length > 0) {
        setFrame((prev) =>
          bundle.manifest.frames.includes(prev) ? prev : bundle.manifest.frames[0]
        );
      }
    } catch {
      setScene(null);
    }
  }, []);

  const shotsLoadedRef = useRef(false);
  const refreshShotsList = useCallback(async () => {
    const res = await api.listShots();
    setShots(res.shots);
    setQueueSummary(res.queue);
    if (res.machine) setMachine(res.machine);
    shotsLoadedRef.current = true;
    return res.shots;
  }, []);

  const machineLabel = machine ? (machine.name ? `${machine.name} · ${machine.ip}` : machine.ip) : "";

  const focusStep2 = useCallback(() => {
    setStep2Highlight(true);
    window.setTimeout(() => {
      step2Ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 50);
    window.setTimeout(() => setStep2Highlight(false), 2200);
  }, []);

  const selectShot = useCallback(
    async (id: string, personId?: number, jumpToStep2?: boolean) => {
      await flushPendingPatch();
      setPlaying(false);
      setUiNotice(null);
      setSeqInterceptError(null);
      setCamError(null);
      setScanError(null);
      setExportError(null);
      let m = await api.getShot(id);
      if (personId !== undefined && m.solved_people.some((sp) => sp.id === personId)) {
        m = await api.updateShot(id, { active_person_ids: [personId] });
      }
      syncLocalFromShot(m, true);
      setDisplayMode((m.work_mode || "human") === "object" ? "mesh" : "both");
      // boxes stay on: solved people are drawn as bodies (no box), the boxes left are the people who can be added
      setShowBoxes(true);
      setShowObjectMasks(!m.has_object_solve_cache);
      const savedView = readRememberedShotViewState(m.id);
      if (savedView?.viewMode === "camera" || savedView?.viewMode === "persp") {
        setViewMode(savedView.viewMode);
      }
      if (m.plate?.frames?.length) {
        const restoreFrame =
          savedView?.frame !== undefined && m.plate.frames.includes(savedView.frame)
            ? savedView.frame
            : m.plate.frames[0];
        setFrame(restoreFrame);
      }
      await loadSceneIfReady(m);
      if (jumpToStep2) {
        focusStep2();
      }
    },
    [flushPendingPatch, syncLocalFromShot, loadSceneIfReady, focusStep2]
  );

  useEffect(() => {
    if (shot?.id) {
      rememberShotViewState(shot.id, frame, viewMode);
    }
  }, [shot?.id, frame, viewMode]);

  // v0.2.12: the open task can leave this computer while it is open - the admin gave it to another computer or
  // deleted it, or another computer moved the unclassified task into its own list.  The list (refreshed every few
  // seconds) shows only what this computer may open: when the open task is no longer in it, the page moves to this
  // computer's own task and says why.  Whether the open task is unclassified follows the list as well.
  const leavingRef = useRef(false);
  useEffect(() => {
    if (!shot?.id || !shotsLoadedRef.current || leavingRef.current) return;
    const item = shots.find((s) => s.id === shot.id);
    if (item) {
      if (!!item.is_public !== !!shot.is_public || (item.owner_label || "") !== (shot.owner_label || "")) {
        setShot((prev) => (prev && prev.id === item.id ? { ...prev, is_public: item.is_public, owner_label: item.owner_label } : prev));
      }
      return;
    }
    leavingRef.current = true;
    (async () => {
      try {
        const fresh = await refreshShotsList();
        if (fresh.some((s) => s.id === shot.id)) return;
        const next = fresh.find((s) => !s.is_public);
        if (next) {
          await selectShot(next.id);
        } else {
          const c = await api.createShot("任务 01");
          await refreshShotsList();
          syncLocalFromShot(c, true);
          setScene(null);
        }
        setUiNotice("刚才打开的任务已不属于本机（被删除，或被移给了另一台电脑），已切换到本机的任务。");
      } catch (err: any) {
        setUiNotice(err?.message || String(err));
      } finally {
        leavingRef.current = false;
      }
    })();
  }, [shots, shot?.id, shot?.is_public, shot?.owner_label, refreshShotsList, selectShot, syncLocalFromShot]);

  useEffect(() => {
    (async () => {
      try {
        const h = await api.health();
        setAppVersion(h.version || "");
        if (!h.initialized) {
          onNavigateAdmin();
          return;
        }
        setOcioOptions(h.ocio);
        setQueueSummary(h.queue);
        const list = await refreshShotsList();
        // the task this browser had open last (when this computer may still open it), else this computer's newest;
        // a public task is opened only when it is the one this browser had open
        const preferredId = readRememberedShotId();
        const targetShot =
          (preferredId ? list.find((s) => s.id === preferredId) : undefined) || list.find((s) => !s.is_public);
        if (targetShot) {
          await selectShot(targetShot.id);
        } else {
          const created = await api.createShot("任务 01");
          await refreshShotsList();
          syncLocalFromShot(created, true);
        }
      } catch (e: any) {
        setUiNotice(e.message || String(e));
      }
    })();
  }, []);

  useEffect(() => {
    const preventWindowDrop = (e: DragEvent) => {
      e.preventDefault();
    };
    window.addEventListener("dragover", preventWindowDrop);
    window.addEventListener("drop", preventWindowDrop);
    return () => {
      window.removeEventListener("dragover", preventWindowDrop);
      window.removeEventListener("drop", preventWindowDrop);
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshShotsList().catch(() => {});
    }, 3000);
    return () => window.clearInterval(timer);
  }, [refreshShotsList]);

  useEffect(() => {
    if (!shot?.id) return;
    const es = new EventSource(`/api/shots/${encodeURIComponent(shot.id)}/events`);
    es.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === "queued" || msg.event === "status") {
          const fresh = await api.getShot(shot.id);
          syncLocalFromShot(fresh);
          await refreshShotsList();
        } else if (msg.event === "stage") {
          setShot((prev) =>
            prev ? { ...prev, status: "running", stage_message: msg.message } : prev
          );
        } else if (msg.event === "progress") {
          setShot((prev) =>
            prev
              ? {
                  ...prev,
                  status: "running",
                  progress: { done: msg.done, total: msg.total, stage: msg.stage || "" },
                }
              : prev
          );
        } else if (msg.event === "partial") {
          // a batch of people is finished while the job goes on: show them now
          const fresh = await api.getShot(shot.id);
          setShot((prev) =>
            prev
              ? {
                  ...prev,
                  solved_people: fresh.solved_people,
                  active_person_ids: fresh.active_person_ids,
                  export_person_ids: fresh.export_person_ids,
                  solved_state: fresh.solved_state,
                  cache_signatures: fresh.cache_signatures,
                  has_view: fresh.has_view,
                  has_solve_cache: fresh.has_solve_cache,
                  has_human_solve_cache: fresh.has_human_solve_cache,
                  solve_warnings: fresh.solve_warnings,
                }
              : prev
          );
          await loadSceneIfReady(fresh);
        } else if (msg.event === "done" || msg.event === "error" || msg.event === "cancelled") {
          const fresh = await api.getShot(shot.id);
          syncLocalFromShot(fresh);
          if (msg.event === "done" && msg.step === "object_solve") {
            setShowObjectMasks(false);
            setDisplayMode("mesh");
          }
          await refreshShotsList();
          await loadSceneIfReady(fresh);
        }
      } catch {
        // ignore
      }
    };
    return () => es.close();
  }, [shot?.id, syncLocalFromShot, refreshShotsList, loadSceneIfReady]);

  const timelineFrames = useMemo(() => {
    if (shot?.plate?.frames?.length) return shot.plate.frames;
    if (scene?.manifest.frames.length) return scene.manifest.frames;
    return [1001];
  }, [shot?.plate, scene]);

  const fps = shot?.plate?.fps || shot?.camera?.summary.fps || 24;

  useEffect(() => {
    if (!playing || timelineFrames.length <= 1) return;
    const intervalMs = 1000 / Math.max(1, fps);
    const timer = window.setInterval(() => {
      setFrame((cur) => {
        const idx = timelineFrames.indexOf(cur);
        const nextIdx = idx < 0 || idx + 1 >= timelineFrames.length ? 0 : idx + 1;
        return timelineFrames[nextIdx];
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, timelineFrames, fps]);

  const autoDetectedPeople = shot?.detection?.people || [];
  const manualEval = useMemo(
    () => buildManualDetectedPeople(manualPeople, timelineFrames),
    [manualPeople, timelineFrames]
  );
  const detectedPeople = useMemo(
    () => [...autoDetectedPeople, ...manualEval.people],
    [autoDetectedPeople, manualEval.people]
  );
  const selectedIds = useMemo(
    () => computeSelectedIds(detectedPeople, selection),
    [detectedPeople, selection]
  );
  const activeManualTrack = useMemo(
    () => manualPeople.find((p) => p.id === activeManualId) || manualPeople[0] || null,
    [manualPeople, activeManualId]
  );

  const isQueued = shot?.status === "queued" || shot?.queue_job?.status === "queued";
  const isRunning = shot?.status === "running" || shot?.queue_job?.status === "running";
  const isBusy = isQueued || isRunning;
  const hasHumanSolve = Boolean(shot?.has_human_solve_cache ?? (shot?.solved_people?.length || 0) > 0);
  const hasObjectSolve = Boolean(shot?.has_object_solve_cache ?? (shot?.solved_objects?.length || 0) > 0);
  const hasSolve = Boolean(shot?.has_solve_cache || hasHumanSolve || hasObjectSolve);
  const hasCamera = Boolean(shot?.camera);
  const canExport = hasSolve && hasCamera;
  const trackedObjects = shot?.object_tracking?.objects || [];
  const hasPositiveObjectPoint = objectPrompts.some((op) =>
    (op.points || []).some((pt) => pt.label === 1)
  );

  const detectCacheValid = useMemo(() => {
    if (!shot?.detection || autoDetectedPeople.length === 0) return false;
    const d = shot.detection;
    if (Math.abs((d.threshold ?? 0.8) - threshold) > 1e-4) return false;
    if (d.grade) {
      const dg = d.grade;
      if (
        Math.abs((dg.exposure ?? 0) - (grade.exposure ?? 0)) > 1e-4 ||
        Math.abs((dg.brightness ?? 0) - (grade.brightness ?? 0)) > 1e-4 ||
        Math.abs((dg.contrast ?? 1) - (grade.contrast ?? 1)) > 1e-4 ||
        Math.abs((dg.saturation ?? 1) - (grade.saturation ?? 1)) > 1e-4 ||
        Math.abs((dg.gamma ?? 1) - (grade.gamma ?? 1)) > 1e-4
      ) {
        return false;
      }
    }
    if (d.ocio_cs && shot.plate?.resolved_colorspace && d.ocio_cs !== shot.plate.resolved_colorspace) {
      return false;
    }
    if (d.ocio_view && shot.plate?.ocio_view && d.ocio_view !== shot.plate.ocio_view) {
      return false;
    }
    return true;
  }, [shot?.detection, shot?.plate, autoDetectedPeople.length, threshold, grade]);

  // People are looked for once per plate. A solve never does it again on its own; a changed grade only makes the
  // existing boxes "older than the picture", which is shown as a hint next to the re-detect button.
  // Boxes written by the v0.1.x detector (no engine tag) are shown but cannot be solved from: the solver needs the
  // per-person outlines of the present tracker, so such a task looks for people anew on the next solve.
  const detectionFromOldVersion =
    Boolean(shot?.detection) && autoDetectedPeople.length > 0 && shot?.detection?.engine !== TRACKER_ENGINE;
  const hasDetection = Boolean(shot?.detection) && autoDetectedPeople.length > 0 && !detectionFromOldVersion;
  const detectionOlderThanGrade = hasDetection && !detectCacheValid;

  const manualOnlyReady = useMemo(() => {
    if (manualEval.people.length === 0) return false;
    const manualIdSet = new Set(manualEval.people.map((p) => p.id));
    if (!shot?.detection || autoDetectedPeople.length === 0) {
      return Array.from(selectedIds).some((id) => manualIdSet.has(id));
    }
    if (selection.mode === "ids" && selectedIds.size > 0) {
      return Array.from(selectedIds).every((id) => manualIdSet.has(id));
    }
    return false;
  }, [manualEval.people, shot?.detection, autoDetectedPeople.length, selection.mode, selectedIds]);

  // Solving adds people: everybody solved before stays, people selected now and not yet solved are added.
  const solvedIdSet = useMemo(() => new Set((shot?.solved_people || []).map((p) => p.id)), [shot?.solved_people]);
  const newSelectedCount = useMemo(
    () => Array.from(selectedIds).filter((id) => !solvedIdSet.has(id)).length,
    [selectedIds, solvedIdSet]
  );

  const humanSolveActionState = useMemo<
    | "detect_and_solve"
    | "solve_selected"
    | "add_people"
    | "add_people_and_redo"
    | "inputs_changed"
    | "fast_postprocess"
    | "up_to_date"
  >(() => {
    if (!hasDetection && !manualOnlyReady) return "detect_and_solve";
    const st = shot?.solved_state;
    if (!hasHumanSolve || !st || !shot?.camera || solvedIdSet.size === 0) return "solve_selected";
    // What makes solved people out of date: another camera, the hand option, or edited manual boxes of a solved
    // person. A new grade does not: a result stays the result of the picture it was estimated from.
    const solvedManual = (list: ManualPersonTrack[]) =>
      JSON.stringify(normalizeManualPeopleList(list).filter((t) => solvedIdSet.has(t.id)));
    const curManNorm = solvedManual(manualPeople);
    const prevManNorm = solvedManual(st.manual_tracks || []);
    const sameNet =
      st.cam_file === (shot.camera.file_path || "") &&
      st.cam_path === (shot.camera.camera_path || "") &&
      st.cam_take === (shot.camera.fbx_take || "") &&
      st.hand_refine === handRefine &&
      curManNorm === prevManNorm;
    const sameSmoothing = Math.abs((st.smoothing ?? 0.5) - smoothing) < 1e-4;
    // every solve brings the people already solved up to the current settings as well, so say so when it will
    if (newSelectedCount > 0) return sameNet && sameSmoothing ? "add_people" : "add_people_and_redo";
    if (!sameNet) return "inputs_changed";
    return sameSmoothing ? "up_to_date" : "fast_postprocess";
  }, [
    hasDetection,
    manualOnlyReady,
    shot?.solved_state,
    shot?.camera,
    hasHumanSolve,
    solvedIdSet,
    newSelectedCount,
    manualPeople,
    handRefine,
    smoothing,
  ]);

  const handleRemoveSolvedPeople = async (ids: number[] | null) => {
    if (!shot) return;
    if (ids === null && !(await confirmAction(`确认清除全部 ${solvedIdSet.size} 个已解算人物？`, "清除"))) return;
    try {
      const updated = await api.removeSolvedPeople(shot.id, ids);
      syncLocalFromShot(updated);
      await loadSceneIfReady(updated);
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleSequenceUpload = async (fileList: File[]) => {
    if (!shot || fileList.length === 0) return;
    try {
      setUiNotice(null);
      setSeqInterceptError(null);
      const checked = validateAndFilterSequenceFiles(fileList);
      setUploadingSeq({ done: 0, total: checked.length });
      const updated = await api.uploadSequenceFiles(shot.id, checked, ocio, fps, (done, total) => {
        setUploadingSeq({ done, total });
      });
      syncLocalFromShot(updated);
      if (updated.plate?.first_frame) setFrame(updated.plate.first_frame);
      setViewMode("camera");
      await refreshShotsList();
      await loadSceneIfReady(updated);
    } catch (e: any) {
      const msg = e?.message || String(e);
      setSeqInterceptError(msg);
      setUiNotice(msg);
    } finally {
      setUploadingSeq(null);
    }
  };

  const handleCameraFileUpload = async (file: File | undefined | null) => {
    if (!file || !shot || isBusy || uploadingCam || updatingCamOpts) return;
    const leaf = cleanClientFilename(file.name);
    if (!isSupported3dAssetFile(leaf)) {
      const msg = `摄影机文件校验失败：不支持的格式 "${leaf}"（仅支持 .abc / .fbx / .usd / .usda / .usdc）`;
      setCamError(msg);
      setUiNotice(msg);
      return;
    }
    if (file.size === 0) {
      const msg = `摄影机文件校验失败：空文件（0 字节）：${leaf}`;
      setCamError(msg);
      setUiNotice(msg);
      return;
    }
    const useUnit = shot.camera?.unit || "m";
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setCamError(null);
      setUploadingCam({ name: leaf, pct: 0, phase: "uploading" });
      const m = await api.uploadCamera(shot.id, file, {
        unit: useUnit,
        onProgress: (pct, phase) => setUploadingCam({ name: leaf, pct, phase }),
      });
      syncLocalFromShot(m);
      await refreshShotsList();
      await loadSceneIfReady(m);
      if ((m.camera?.listing?.cameras?.length || 0) > 1) {
        setShowCamTreeModal(true);
      }
    } catch (err: any) {
      const msg = err?.message || String(err);
      setCamError(msg);
      setUiNotice(msg);
    } finally {
      setUploadingCam(null);
    }
  };

  const handleUpdateCameraOpts = async (opts: {
    camera_path?: string;
    unit?: string;
    fbx_take?: string;
    frame_offset?: number;
  }) => {
    if (!shot?.camera || isBusy || updatingCamOpts) return;
    try {
      setUiNotice(null);
      setCamError(null);
      setUpdatingCamOpts(true);
      const m = await api.updateCameraOptions(shot.id, {
        camera_path: opts.camera_path ?? shot.camera.camera_path,
        unit: opts.unit ?? (shot.camera.unit || "m"),
        fbx_take: opts.fbx_take ?? shot.camera.fbx_take,
        frame_offset: opts.frame_offset ?? shot.camera.frame_offset,
      });
      syncLocalFromShot(m);
      await refreshShotsList();
      await loadSceneIfReady(m);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setCamError(msg);
      setUiNotice(msg);
    } finally {
      setUpdatingCamOpts(false);
    }
  };

  const handleSceneScanFileUpload = async (file: File | undefined | null) => {
    if (!file || !shot || isBusy || uploadingScan || updatingScanOpts) return;
    const leaf = cleanClientFilename(file.name);
    if (!isSupported3dAssetFile(leaf)) {
      const msg = `场景扫描文件校验失败：不支持的格式 "${leaf}"（仅支持 .abc / .fbx / .usd / .usda / .usdc）`;
      setScanError(msg);
      setUiNotice(msg);
      return;
    }
    if (file.size === 0) {
      const msg = `场景扫描文件校验失败：空文件（0 字节）：${leaf}`;
      setScanError(msg);
      setUiNotice(msg);
      return;
    }
    const defaultScanUnit =
      shot.scene_scan?.unit || (shot.camera?.unit === "cm" ? "cm" : "m");
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setScanError(null);
      setUploadingScan({ name: leaf, pct: 0, phase: "uploading" });
      const m = await api.uploadSceneScan(shot.id, file, {
        unit: defaultScanUnit,
        onProgress: (pct, phase) => setUploadingScan({ name: leaf, pct, phase }),
      });
      syncLocalFromShot(m);
      await refreshShotsList();
      await loadSceneIfReady(m);
      if ((m.scene_scan?.listing?.meshes?.length || 0) > 1) {
        setShowScanTreeModal(true);
      }
    } catch (err: any) {
      const msg = err?.message || String(err);
      setScanError(msg);
      setUiNotice(msg);
    } finally {
      setUploadingScan(null);
    }
  };

  const handleUpdateSceneScanOpts = async (opts: {
    unit?: string;
    selected_paths?: string[];
  }) => {
    if (!shot?.scene_scan || isBusy || updatingScanOpts) return;
    try {
      setUiNotice(null);
      setScanError(null);
      setUpdatingScanOpts(true);
      const m = await api.updateSceneScanOptions(shot.id, {
        unit: opts.unit ?? (shot.scene_scan.unit || "m"),
        selected_paths: opts.selected_paths ?? shot.scene_scan.selected_paths,
      });
      syncLocalFromShot(m);
      await refreshShotsList();
      await loadSceneIfReady(m);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setScanError(msg);
      setUiNotice(msg);
    } finally {
      setUpdatingScanOpts(false);
    }
  };

  // Step 4: "按扫描对齐" / "对齐后" (true) and "对齐前" (false).  The server keeps the results of both states,
  // so this either switches at once or queues a job for the people who have no up-to-date result in that state.
  const handleScanAlign = async (enabled: boolean) => {
    if (!shot?.scene_scan || isBusy || uploadingScan || updatingScanOpts || switchingScanAlign) return;
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setScanError(null);
      setSwitchingScanAlign(true);
      const m = await api.setScanAlignment(shot.id, enabled);
      syncLocalFromShot(m);
      await refreshShotsList();
      await loadSceneIfReady(m);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setScanError(msg);
      setUiNotice(msg);
    } finally {
      setSwitchingScanAlign(false);
    }
  };

  const makeDropProps = (
    targetKey: string,
    onDroppedFiles: (files: File[]) => Promise<void> | void,
    disabled = false
  ) => ({
    onDragEnter: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) {
        e.dataTransfer.dropEffect = "copy";
        setDragHoverTarget(targetKey);
      }
    },
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) {
        e.dataTransfer.dropEffect = "copy";
        if (dragHoverTarget !== targetKey) setDragHoverTarget(targetKey);
      }
    },
    onDragLeave: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const rel = e.relatedTarget as Node | null;
      if (!rel || !e.currentTarget.contains(rel)) {
        setDragHoverTarget((cur) => (cur === targetKey ? null : cur));
      }
    },
    onDrop: async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragHoverTarget(null);
      if (disabled) return;
      const dropped = await collectDroppedFiles(e.dataTransfer);
      await onDroppedFiles(dropped);
    },
  });

  const handleChangeOcio = async (next: OcioSettings) => {
    setOcio(next);
    if (!shot?.plate) return;
    try {
      setUpdatingOcio(true);
      setUiNotice(null);
      const updated = await api.updateOcio(shot.id, next);
      syncLocalFromShot(updated);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    } finally {
      setUpdatingOcio(false);
    }
  };

  const handleRunDetect = async () => {
    if (!shot) return;
    const manualIds = new Set(manualPeople.map((t) => t.id));
    const trackedSolved = (shot.solved_people || []).filter((sp) => !manualIds.has(sp.id)).length;
    if (
      trackedSolved > 0 &&
      !(await confirmAction(`重新检测后人物编号将重新分配，已解算的 ${trackedSolved} 人将被清除。是否继续？`, "继续"))
    ) {
      return;
    }
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setShowBoxes(true);
      const res = await api.runDetect(shot.id, threshold, grade);
      syncLocalFromShot(res.shot);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleRunSolve = async () => {
    if (!shot) return;
    if (!shot.camera) {
      const msg = "请先在第 2 步导入跟踪摄影机";
      setCamError(msg);
      setUiNotice(msg);
      focusStep2();
      return;
    }
    if (humanSolveActionState === "detect_and_solve" && detectionFromOldVersion) {
      const manualIds = new Set(manualPeople.map((t) => t.id));
      const trackedSolved = (shot.solved_people || []).filter((sp) => !manualIds.has(sp.id)).length;
      if (
        trackedSolved > 0 &&
        !(await confirmAction(
          `该任务的人物选框由旧版本生成，解算前需重新检测：人物编号将重新分配，已解算的 ${trackedSolved} 人将被清除。是否继续？`,
          "继续"
        ))
      ) {
        return;
      }
    }
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setExportError(null);
      setManualDrawMode(false);
      const res = await api.runSolve(shot.id, {
        grade,
        detect_params: { threshold },
        manual_people: normalizeManualPeopleList(manualPeople),
        selection,
        solve_params: { hand_refine: handRefine, use_mask: useMask, lock_shape: lockShape, smoothing },
      });
      syncLocalFromShot(res.shot);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const commitManualPeople = useCallback(
    (nextRaw: ManualPersonTrack[]) => {
      const norm = normalizeManualPeopleList(nextRaw);
      setManualPeople(norm);
      scheduleSavePatch({ manual_people: norm });
      return norm;
    },
    [scheduleSavePatch]
  );

  const handleAddManualPerson = () => {
    const nextId = Math.max(101, ...manualPeople.map((p) => p.id + 1));
    const nextTrack: ManualPersonTrack = {
      id: nextId,
      name: `手动 #${nextId}`,
      enabled: true,
      keyframes: [],
    };
    commitManualPeople([...manualPeople, nextTrack]);
    setActiveManualId(nextId);
    setManualDrawMode(true);
    setShowBoxes(true);
    setWipeEnabled(false);
    setViewMode("camera");
    if (selection.mode === "ids") {
      const curTokens = (selection.ids || "")
        .split(/[,;\s]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      if (!curTokens.includes(String(nextId))) {
        const nextIds = [...curTokens, String(nextId)].join(",");
        updateSelection({ ...selection, ids: nextIds });
      }
    }
  };

  const handleDrawManualBox = useCallback(
    (f: number, box: [number, number, number, number]) => {
      setShowBoxes(true);
      setManualPeople((prev) => {
        let targetId = activeManualId;
        let base = prev;
        if (!targetId || !prev.some((p) => p.id === targetId)) {
          targetId = Math.max(101, ...prev.map((p) => p.id + 1));
          base = [...prev, { id: targetId, name: `手动 #${targetId}`, enabled: true, keyframes: [] }];
          setActiveManualId(targetId);
        }
        const updated = normalizeManualPeopleList(
          base.map((trk) =>
            trk.id === targetId
              ? upsertManualKeyframe({ ...trk, enabled: true }, { frame: f, status: "active", box })
              : trk
          )
        );
        scheduleSavePatch({ manual_people: updated });
        return updated;
      });
    },
    [activeManualId, scheduleSavePatch]
  );

  const handleSetManualStatusAtFrame = (status: "occluded" | "off") => {
    if (!activeManualTrack) return;
    const next = manualPeople.map((trk) =>
      trk.id === activeManualTrack.id
        ? upsertManualKeyframe(trk, { frame, status })
        : trk
    );
    commitManualPeople(next);
  };

  const handlePromoteInterpolatedToKeyframe = () => {
    if (!activeManualTrack) return;
    const mp = manualEval.people.find((p) => p.id === activeManualTrack.id);
    const curBox = mp?.boxes[String(frame)];
    if (!curBox) return;
    const next = manualPeople.map((trk) =>
      trk.id === activeManualTrack.id
        ? upsertManualKeyframe(trk, { frame, status: "active", box: curBox })
        : trk
    );
    commitManualPeople(next);
  };

  const handleDeleteManualKeyframeAtCurrentFrame = () => {
    if (!activeManualTrack) return;
    const next = manualPeople.map((trk) =>
      trk.id === activeManualTrack.id ? removeManualKeyframeAt(trk, frame) : trk
    );
    commitManualPeople(next);
  };

  const handleRemoveManualPerson = (pid: number) => {
    const next = manualPeople.filter((trk) => trk.id !== pid);
    const norm = commitManualPeople(next);
    if (activeManualId === pid) {
      setActiveManualId(norm[0]?.id ?? null);
    }
    if (norm.length === 0) {
      setManualDrawMode(false);
    }
  };

  const handleJumpManualKeyframe = (dir: -1 | 1) => {
    if (!activeManualTrack || activeManualTrack.keyframes.length === 0) return;
    const kfs = activeManualTrack.keyframes;
    setPlaying(false);
    if (dir < 0) {
      for (let i = kfs.length - 1; i >= 0; i--) {
        if (kfs[i].frame < frame) {
          setFrame(kfs[i].frame);
          return;
        }
      }
      setFrame(kfs[0].frame);
    } else {
      for (let i = 0; i < kfs.length; i++) {
        if (kfs[i].frame > frame) {
          setFrame(kfs[i].frame);
          return;
        }
      }
      setFrame(kfs[kfs.length - 1].frame);
    }
  };

  const handleCancelTask = async () => {
    if (!shot) return;
    try {
      const m = await api.cancelShot(shot.id);
      syncLocalFromShot(m);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const updateGrade = useCallback(
    (next: GradeParams) => {
      setGrade(next);
      scheduleSavePatch({ grade: next });
    },
    [scheduleSavePatch]
  );

  const updateSelection = useCallback(
    (next: SelectionState) => {
      setSelection(next);
      scheduleSavePatch({ selection: next });
    },
    [scheduleSavePatch]
  );

  const handleToggleBoxPerson = (pid: number) => {
    const cur = new Set(selectedIds);
    if (cur.has(pid) && cur.size > 1) cur.delete(pid);
    else cur.add(pid);
    const nextIds = Array.from(cur).sort((a, b) => a - b).join(",");
    updateSelection({ ...selection, mode: "ids", ids: nextIds || String(pid) });
  };

  // Which solved people go into the exported files.  Stored with the task; an empty list means everybody.
  const solvedIdList = useMemo(() => (shot?.solved_people || []).map((p) => p.id), [shot?.solved_people]);
  const exportIdSet = useMemo(() => {
    const chosen = shot?.export_person_ids || [];
    return new Set<number>(chosen.length ? chosen : solvedIdList);
  }, [shot?.export_person_ids, solvedIdList]);

  const saveExportPeople = async (ids: number[]) => {
    if (!shot || ids.length === 0) return;
    try {
      const updated = await api.updateShot(shot.id, { export_person_ids: ids });
      syncLocalFromShot(updated);
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleToggleExportPerson = (pid: number) => {
    const cur = new Set(exportIdSet);
    if (cur.has(pid)) {
      if (cur.size > 1) cur.delete(pid); // at least one person stays chosen
    } else {
      cur.add(pid);
    }
    saveExportPeople(Array.from(cur).sort((p, q) => p - q));
  };

  const handleSwitchWorkMode = async (nextMode: WorkMode) => {
    setWorkMode(nextMode);
    setDisplayMode(nextMode === "object" ? "mesh" : "both");
    if (nextMode === "object" && viewMode !== "camera" && !hasObjectSolve) {
      setViewMode("camera");
    }
    if (shot) {
      api.updateShot(shot.id, { work_mode: nextMode }).catch(() => {});
    }
  };

  const handlePickObjectPoint = (f: number, x: number, y: number, label: 0 | 1) => {
    setShowObjectMasks(true);
    setObjectPrompts((prev) => {
      const next = prev.map((op) =>
        op.id === activeObjectId
          ? { ...op, points: [...(op.points || []), { frame: f, x, y, label }] }
          : op
      );
      if (shot) {
        api.updateShot(shot.id, { object_prompts: next }).catch(() => {});
      }
      return next;
    });
  };

  const handleAddNewObject = () => {
    const maxId = objectPrompts.reduce((m, o) => Math.max(m, o.id), 0);
    const nextId = maxId + 1;
    const next = [
      ...objectPrompts,
      { id: nextId, name: `object_${String(nextId).padStart(2, "0")}`, points: [] },
    ];
    setObjectPrompts(next);
    setActiveObjectId(nextId);
    if (shot) {
      api.updateShot(shot.id, { object_prompts: next }).catch(() => {});
    }
  };

  const handleClearObjectPoints = (oid: number) => {
    const next = objectPrompts.map((op) => (op.id === oid ? { ...op, points: [] } : op));
    setObjectPrompts(next);
    if (shot) {
      api.updateShot(shot.id, { object_prompts: next }).catch(() => {});
    }
  };

  const handleRemoveObjectPrompt = (oid: number) => {
    if (objectPrompts.length <= 1) {
      handleClearObjectPoints(oid);
      return;
    }
    const next = objectPrompts.filter((op) => op.id !== oid);
    setObjectPrompts(next);
    if (activeObjectId === oid) {
      setActiveObjectId(next[0].id);
    }
    if (shot) {
      api.updateShot(shot.id, { object_prompts: next }).catch(() => {});
    }
  };

  const handleRunObjectTrack = async () => {
    if (!shot) return;
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setShowObjectMasks(true);
      const res = await api.runObjectTrack(shot.id, objectPrompts, grade);
      syncLocalFromShot(res.shot);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleRunObjectSolve = async () => {
    if (!shot) return;
    if (!shot.camera) {
      const msg = "请先在第 2 步导入跟踪摄影机";
      setCamError(msg);
      setUiNotice(msg);
      focusStep2();
      return;
    }
    try {
      await flushPendingPatch();
      setUiNotice(null);
      setExportError(null);
      const res = await api.runObjectSolve(shot.id, {
        object_prompts: objectPrompts,
        object_solve_params: { smoothing: objectSmoothing },
        grade,
      });
      syncLocalFromShot(res.shot);
      await refreshShotsList();
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleToggleActiveSolvedObject = async (oid: number) => {
    if (!shot || !shot.solved_objects) return;
    const cur = new Set(
      shot.active_object_ids?.length
        ? shot.active_object_ids
        : shot.solved_objects.map((o) => o.id)
    );
    if (cur.has(oid) && cur.size > 1) cur.delete(oid);
    else cur.add(oid);
    const next = Array.from(cur).sort((a, b) => a - b);
    try {
      const updated = await api.updateShot(shot.id, { active_object_ids: next });
      syncLocalFromShot(updated);
      await loadSceneIfReady(updated);
    } catch (e: any) {
      setUiNotice(e.message);
    }
  };

  const handleChangeExportUnit = (nextUnit: "m" | "cm") => {
    setExportUnit(nextUnit);
  };

  // seconds a file has been in the making (shown on the button; big point caches take a while)
  const [buildSeconds, setBuildSeconds] = useState(0);
  useEffect(() => {
    if (!downloadingFmt) {
      setBuildSeconds(0);
      return;
    }
    const t0 = Date.now();
    const timer = window.setInterval(() => setBuildSeconds(Math.floor((Date.now() - t0) / 1000)), 500);
    return () => window.clearInterval(timer);
  }, [downloadingFmt]);

  const handleDirectDownload = async (fmt: string) => {
    if (!shot || !canExport || downloadingFmt) return;
    try {
      setDownloadingFmt(fmt);
      setExportError(null);
      setUiNotice(null);
      const updated = await api.downloadSingleFormat(shot.id, fmt, exportUnit);
      syncLocalFromShot(updated);
    } catch (e: any) {
      const msg = e?.message || String(e);
      setExportError(msg);
      setUiNotice(msg);
    } finally {
      setDownloadingFmt(null);
    }
  };

  const solvedFramesSet = useMemo(() => {
    const s = new Set<number>();
    scene?.characters.forEach((c) => {
      c.frames.forEach((f, idx) => {
        if (!c.visible || c.visible[idx] > 0) s.add(f);
      });
    });
    scene?.objects?.forEach((o) => {
      o.frames.forEach((f, idx) => {
        if (!o.visible || o.visible[idx] > 0) s.add(f);
      });
    });
    return s;
  }, [scene]);

  const plateRev = `${shot?.plate?.source_name || ""}:${shot?.plate?.frame_count || 0}:${shot?.plate?.resolved_colorspace || "auto"}:${shot?.plate?.ocio_view || "aces_sdr"}`;

  return (
    <div className="app-shell">
      {/* Top Bar — Clean Brand + GPU Status + Admin Entry */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-title">入镜 · ShotSpace</span>
          {appVersion && (
            <span className="brand-version" title="当前运行的版本">
              v{appVersion}
            </span>
          )}
        </div>

        {/* While a job of this task computes: a cup of coffee and a word, in the empty middle of the bar.  It is
            laid over the bar (nothing moves when it comes or goes) and never takes the pointer. */}
        <div className={`coffee-note ${isRunning ? "is-on" : ""}`} role="status" aria-hidden={!isRunning}>
          <CoffeeCup />
          <span>正在计算~喝杯咖啡休息一会儿~~~</span>
        </div>

        <div className="topbar-right">
          {machineLabel && (
            <span className="topbar-machine" title="本机。任务按电脑区分：这里只能看到和打开本机建的任务">
              本机 {machineLabel}
            </span>
          )}
          {queueSummary && (
            <span
              className={`tag ${
                queueSummary.all_gpus_disabled
                  ? "tag-warn"
                  : queueSummary.running_count > 0 || queueSummary.queued_count > 0
                  ? "tag-info"
                  : "tag-ok"
              }`}
            >
              {queueSummary.all_gpus_disabled
                ? `显卡均已停用 · 排队 ${queueSummary.queued_count}`
                : `显卡 ${queueSummary.enabled_gpu_count}/${queueSummary.total_gpu_count} · 排队 ${queueSummary.queued_count} · 运行 ${queueSummary.running_count}`}
            </span>
          )}
          <button type="button" className="btn btn-sm" onClick={onNavigateAdmin}>
            管理后台
          </button>
        </div>
      </header>

      {/* Main Split Workbench */}
      <div className="workbench">
        <aside className="sidebar">
          {/* Left Full-Width Task & Character Switcher + New Task Card */}
          <section className="step-card">
            <button
              type="button"
              className="task-trigger-btn"
              onClick={() => setShowGalleryModal(true)}
              title="打开人物与任务选择面板"
            >
              {shot?.plate ? (
                <img
                  src={`/api/shots/${encodeURIComponent(shot.id)}/thumbnail`}
                  alt=""
                  className="task-trigger-thumb"
                />
              ) : (
                <span className="task-trigger-dot" />
              )}
              <span className="task-trigger-name" title={shot?.name || undefined}>
                {shot?.name || "选择任务"}
              </span>
              {hasSolve ? (
                <span className="tag tag-ok">
                  {shot?.camera ? "已对齐摄影机" : "已解算"}
                </span>
              ) : (
                <span className="tag tag-muted">未解算</span>
              )}
              <span className="task-trigger-more">
                人物与任务列表
                <IconChevronDown size={12} className="icon-trail" />
              </span>
            </button>

            {shot?.is_public && (
              <div className="task-public-row">
                <span className="tag tag-warn">未分类任务</span>
                <span className="hint">升级前建的、未能确定属于哪台电脑的任务，所有电脑都能看到。</span>
                <button
                  type="button"
                  className={`btn btn-sm ${claimingShot ? "is-busy" : ""}`}
                  disabled={claimingShot}
                  onClick={async () => {
                    if (!shot) return;
                    setClaimingShot(true);
                    try {
                      const m = await api.claimShot(shot.id);
                      syncLocalFromShot(m);
                      await refreshShotsList();
                    } catch (err: any) {
                      setUiNotice(err?.message || String(err));
                    } finally {
                      setClaimingShot(false);
                    }
                  }}
                >
                  {claimingShot && <span className="spinner" />}
                  移到我的任务
                </button>
              </div>
            )}

            <button type="button" className="btn btn-block" onClick={() => setShowNewTaskModal(true)}>
              + 新建任务
            </button>
          </section>

          {/* a notice that is already shown inside its own step is not repeated here */}
          {uiNotice &&
            uiNotice !== seqInterceptError &&
            uiNotice !== camError &&
            uiNotice !== scanError &&
            uiNotice !== exportError && (
              <div className="alert alert-error">
                <div className="alert-row">
                  <span>{uiNotice}</span>
                  <button type="button" className="btn btn-xs" onClick={() => setUiNotice(null)}>
                    关闭
                  </button>
                </div>
              </div>
            )}

          {shot?.error && <div className="alert alert-error">{shot.error}</div>}

          {/* STEP 1: Input Image Sequence + OCIO + Real-Time Grade Preview */}
          <section
            className={`step-card ${dragHoverTarget === "step1-card" ? "step-card-drop-active" : ""}`}
            {...makeDropProps("step1-card", handleSequenceUpload, isBusy || !!uploadingSeq)}
          >
            <div className="step-header">
              <div className="step-title">
                <span className="step-num">1</span>
                <span>序列帧与调色</span>
              </div>
              <div className="step-actions">
                <input
                  ref={folderInputRef}
                  type="file"
                  multiple
                  {...({ webkitdirectory: "", directory: "" } as any)}
                  style={{ display: "none" }}
                  onChange={async (e) => {
                    const list = Array.from(e.target.files || []);
                    e.target.value = "";
                    await handleSequenceUpload(list);
                  }}
                />
                <input
                  ref={filesInputRef}
                  type="file"
                  multiple
                  accept=".exr,.dpx,.cin,.tif,.tiff,.png,.jpg,.jpeg,.tga,.bmp,.webp,.hdr,.zip,.tar,.tgz"
                  style={{ display: "none" }}
                  onChange={async (e) => {
                    const list = Array.from(e.target.files || []);
                    e.target.value = "";
                    await handleSequenceUpload(list);
                  }}
                />
                <button
                  type="button"
                  className={`btn btn-sm btn-drop-target ${shot?.plate ? "" : "btn-primary"} ${
                    dragHoverTarget === "btn-seq-folder" ? "btn-drop-active" : ""
                  }`}
                  disabled={isBusy || !!uploadingSeq}
                  title="选择序列帧文件夹，或将文件夹拖拽至此"
                  onClick={() => folderInputRef.current?.click()}
                  {...makeDropProps("btn-seq-folder", handleSequenceUpload, isBusy || !!uploadingSeq)}
                >
                  选择文件夹
                </button>
                <button
                  type="button"
                  className={`btn btn-sm btn-drop-target ${
                    dragHoverTarget === "btn-seq-files" ? "btn-drop-active" : ""
                  }`}
                  disabled={isBusy || !!uploadingSeq}
                  title="选择序列帧文件或 .zip 压缩包，或将文件拖拽至此"
                  onClick={() => filesInputRef.current?.click()}
                  {...makeDropProps("btn-seq-files", handleSequenceUpload, isBusy || !!uploadingSeq)}
                >
                  选择文件
                </button>
              </div>
            </div>

            {seqInterceptError && (
              <div className="alert alert-error">
                <div className="alert-row">
                  <span>{seqInterceptError}</span>
                  <button
                    type="button"
                    className="btn btn-xs"
                    onClick={() => {
                      setSeqInterceptError(null);
                      setUiNotice(null);
                    }}
                  >
                    关闭
                  </button>
                </div>
              </div>
            )}

            {uploadingSeq ? (
              <div className="meta-pill is-active">
                <div className="progress-head">
                  <span>正在检测序列帧并执行 OCIO 转换...</span>
                  <strong className="tnum">
                    {uploadingSeq.done} / {uploadingSeq.total}
                  </strong>
                </div>
                <div className="progress">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.round((uploadingSeq.done / Math.max(1, uploadingSeq.total)) * 100)}%` }}
                  />
                </div>
              </div>
            ) : shot?.plate ? (
              <div className="meta-pill">
                <strong>{shot.plate.source_name}</strong> · {shot.plate.width}×{shot.plate.height} ({shot.plate.bit_depth}) · 帧{" "}
                <strong>
                  {shot.plate.first_frame}–{shot.plate.last_frame}
                </strong>{" "}
                （{shot.plate.frame_count} 帧 · 连续）
              </div>
            ) : (
              <div className="meta-pill">支持 .exr / .dpx / .tif / .png / .jpg / .zip</div>
            )}

            {(shot?.plate?.warnings || []).map((w, idx) => (
              <div key={idx} className="alert alert-warn">
                {w}
              </div>
            ))}

            <div className="ocio-inline-row">
              <div className="ocio-inline-item">
                <label>色彩</label>
                <select
                  disabled={isBusy || updatingOcio}
                  value={ocio.input_colorspace}
                  title="OCIO 输入色彩空间"
                  onChange={(e) => handleChangeOcio({ ...ocio, input_colorspace: e.target.value })}
                >
                  {(ocioOptions?.colorspaces || [
                    { id: "auto", label: "自动 (.exr=ACEScg)" },
                    { id: "ACEScg", label: "ACEScg" },
                    { id: "sRGB Encoded Rec.709 (sRGB)", label: "sRGB" },
                  ]).map((cs) => (
                    <option key={cs.id} value={cs.id}>
                      {cs.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="ocio-inline-item">
                <label>映射</label>
                <select
                  disabled={isBusy || updatingOcio}
                  value={ocio.ocio_view}
                  title="OCIO sRGB 显示映射"
                  onChange={(e) => handleChangeOcio({ ...ocio, ocio_view: e.target.value })}
                >
                  {(ocioOptions?.views || [
                    { id: "aces_sdr", label: "ACES 1.0 SDR" },
                    { id: "un_tone_mapped", label: "sRGB（无色调映射）" },
                  ]).map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grade-compact-grid">
              <div className="grade-cell">
                <label>曝光</label>
                <Range
                  aria-label="曝光"
                  min={-4}
                  max={4}
                  step={0.05}
                  value={grade.exposure}
                  onChange={(e) => updateGrade({ ...grade, exposure: Number(e.target.value) })}
                />
                <span className="grade-val">
                  {grade.exposure >= 0 ? `+${grade.exposure.toFixed(2)}` : grade.exposure.toFixed(2)}
                </span>
              </div>

              <div className="grade-cell">
                <label>对比</label>
                <Range
                  aria-label="对比"
                  min={0.2}
                  max={2.5}
                  step={0.02}
                  value={grade.contrast ?? 1.0}
                  onChange={(e) => updateGrade({ ...grade, contrast: Number(e.target.value) })}
                />
                <span className="grade-val">{(grade.contrast ?? 1.0).toFixed(2)}</span>
              </div>

              <div className="grade-cell">
                <label>亮度</label>
                <Range
                  aria-label="亮度"
                  min={-0.5}
                  max={0.5}
                  step={0.01}
                  value={grade.brightness}
                  onChange={(e) => updateGrade({ ...grade, brightness: Number(e.target.value) })}
                />
                <span className="grade-val">{grade.brightness.toFixed(2)}</span>
              </div>

              <div className="grade-cell">
                <label>饱和</label>
                <Range
                  aria-label="饱和"
                  min={0.0}
                  max={2.5}
                  step={0.02}
                  value={grade.saturation ?? 1.0}
                  onChange={(e) => updateGrade({ ...grade, saturation: Number(e.target.value) })}
                />
                <span className="grade-val">{(grade.saturation ?? 1.0).toFixed(2)}</span>
              </div>

              <div className="grade-cell">
                <label>Gamma</label>
                <Range
                  aria-label="Gamma"
                  min={0.2}
                  max={3.0}
                  step={0.02}
                  value={grade.gamma}
                  onChange={(e) => updateGrade({ ...grade, gamma: Number(e.target.value) })}
                />
                <span className="grade-val">{grade.gamma.toFixed(2)}</span>
              </div>

              <div className="grade-actions">
                <button
                  type="button"
                  className="btn btn-sm btn-grow"
                  title="按住以显示未调色的原片"
                  onMouseDown={() => setBypassGrade(true)}
                  onMouseUp={() => setBypassGrade(false)}
                  onMouseLeave={() => setBypassGrade(false)}
                >
                  对比原片
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() =>
                    updateGrade({ exposure: 0, brightness: 0, contrast: 1, saturation: 1, gamma: 1 })
                  }
                >
                  重置
                </button>
              </div>
            </div>
          </section>

          {/* STEP 2: Import Tracked Camera ONLY (Mandatory Shot Input) */}
          <section
            ref={step2Ref}
            className={`step-card ${step2Highlight ? "step-card-highlight" : ""} ${
              dragHoverTarget === "step2-cam-card" ? "step-card-drop-active" : ""
            }`}
            {...makeDropProps(
              "step2-cam-card",
              (files) => handleCameraFileUpload(files[0]),
              isBusy || !!uploadingCam || updatingCamOpts
            )}
          >
            <div className="step-header">
              <div className="step-title">
                <span className="step-num">2</span>
                <span>导入跟踪摄影机</span>
              </div>
              <div className="step-actions">
                <input
                  ref={camInputRef}
                  type="file"
                  accept=".abc,.fbx,.usd,.usda,.usdc,.usdz"
                  style={{ display: "none" }}
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    e.target.value = "";
                    await handleCameraFileUpload(f);
                  }}
                />
                <button
                  type="button"
                  className={`btn btn-sm btn-drop-target ${shot?.plate && !shot?.camera ? "btn-primary" : ""} ${
                    dragHoverTarget === "btn-cam-upload" ? "btn-drop-active" : ""
                  }`}
                  disabled={isBusy || !!uploadingCam || updatingCamOpts}
                  title="选择摄影机文件（.abc / .fbx / .usd），或将文件拖拽至此"
                  onClick={() => camInputRef.current?.click()}
                  {...makeDropProps(
                    "btn-cam-upload",
                    (files) => handleCameraFileUpload(files[0]),
                    isBusy || !!uploadingCam || updatingCamOpts
                  )}
                >
                  {uploadingCam
                    ? "正在解析摄影机..."
                    : updatingCamOpts
                    ? "正在更新..."
                    : shot?.camera
                    ? "更换摄影机"
                    : "导入摄影机"}
                </button>
                {shot?.camera && (
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    disabled={isBusy || !!uploadingCam || updatingCamOpts}
                    onClick={async () => {
                      if (!shot) return;
                      setCamError(null);
                      const m = await api.clearCamera(shot.id);
                      syncLocalFromShot(m);
                      await refreshShotsList();
                      await loadSceneIfReady(m);
                    }}
                  >
                    移除
                  </button>
                )}
              </div>
            </div>

            {camError && (
              <div className="alert alert-error">
                <div className="alert-row">
                  <span>{camError}</span>
                  <button
                    type="button"
                    className="btn btn-xs"
                    onClick={() => {
                      setCamError(null);
                      setUiNotice(null);
                    }}
                  >
                    关闭
                  </button>
                </div>
              </div>
            )}

            {(uploadingCam || updatingCamOpts) && (
              <div className="meta-pill is-active">
                <div className="progress-head">
                  <span>
                    {uploadingCam
                      ? uploadingCam.phase === "uploading"
                        ? `正在上传摄影机文件：${uploadingCam.name}`
                        : `正在解析摄影机层级与内参：${uploadingCam.name}...`
                      : "正在切换摄影机节点/单位..."}
                  </span>
                  {uploadingCam && uploadingCam.phase === "uploading" && (
                    <strong className="tnum">{uploadingCam.pct}%</strong>
                  )}
                </div>
                <div className="progress">
                  {uploadingCam && uploadingCam.phase === "uploading" ? (
                    <div className="progress-fill" style={{ width: `${uploadingCam.pct}%` }} />
                  ) : (
                    <div className="progress-indeterminate" />
                  )}
                </div>
              </div>
            )}

            {shot?.camera ? (
              <div className="meta-pill is-stack">
                <div>
                  <strong>{shot.camera.summary.name}</strong> ({shot.camera.listing.format.toUpperCase()}) · 焦距{" "}
                  <strong>
                    {Array.isArray(shot.camera.summary.focal_mm)
                      ? `${shot.camera.summary.focal_mm[0]}–${shot.camera.summary.focal_mm[1]} mm`
                      : `${shot.camera.summary.focal_mm} mm`}
                  </strong>{" "}
                  · 片门 {shot.camera.summary.h_aperture_mm}×{shot.camera.summary.v_aperture_mm} mm
                </div>
                <div>
                  帧 {shot.camera.summary.frames[0]}–{shot.camera.summary.frames[1]} · 位移{" "}
                  <strong>
                    {shot.camera.summary.travel_cm !== undefined
                      ? shot.camera.summary.travel_cm >= 1.0
                        ? `${(shot.camera.summary.travel_cm / 100).toFixed(2)} m`
                        : "静止"
                      : "—"}
                  </strong>{" "}
                  · Y-Up
                  {hasSolve ? " · 已就绪" : ""}
                </div>

                <div className="form-row">
                  <span className="form-label">摄影机单位</span>
                  <select
                    className="form-select"
                    disabled={updatingCamOpts || !!uploadingCam}
                    value={shot.camera.unit || "m"}
                    onChange={(e) => handleUpdateCameraOpts({ unit: e.target.value })}
                  >
                    <option value="m">米 (m) · 默认</option>
                    <option value="cm">厘米 (cm)</option>
                    <option value="auto">按文件头元数据</option>
                  </select>
                </div>

                <div className="form-row">
                  <span className="form-label">层级节点</span>
                  <button
                    type="button"
                    className={`btn btn-sm btn-split ${shot.camera.listing.cameras.length > 1 ? "is-on" : ""}`}
                    disabled={updatingCamOpts || !!uploadingCam}
                    onClick={() => setShowCamTreeModal(true)}
                    title={shot.camera.camera_path}
                  >
                    <span className="btn-split-value">
                      <IconCamera size={14} />
                      <span>{shot.camera.camera_path}</span>
                    </span>
                    <span className="btn-split-more">
                      {shot.camera.listing.cameras.length > 1
                        ? `切换 (${shot.camera.listing.cameras.length}台)`
                        : "查看层级"}
                      <IconChevronDown size={12} className="icon-trail" />
                    </span>
                  </button>
                </div>

                {shot.camera.listing.takes.length > 1 && (
                  <div className="form-row">
                    <span className="form-label">动画层</span>
                    <select
                      className="form-select"
                      disabled={updatingCamOpts || !!uploadingCam}
                      value={shot.camera.fbx_take || shot.camera.listing.auto_take}
                      onChange={(e) => handleUpdateCameraOpts({ fbx_take: e.target.value })}
                    >
                      {shot.camera.listing.takes.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {shot.camera.summary.travel_cm !== undefined &&
                  shot.camera.summary.frame_count > 1 &&
                  shot.camera.summary.travel_cm > 0.05 &&
                  shot.camera.summary.travel_cm < 12.0 && (
                    <div className="alert alert-warn">
                      当前摄影机总位移仅 {(shot.camera.summary.travel_cm / 100).toFixed(3)} m（
                      {shot.camera.summary.travel_cm.toFixed(1)} cm），如原始资产为米制，请将「摄影机单位」设为「米 (m)」。
                    </div>
                  )}
              </div>
            ) : !uploadingCam ? (
              <div className="meta-pill">支持 .abc / .fbx / .usd，默认单位：米</div>
            ) : null}

            {(shot?.alignment_report?.warnings || []).map((w, idx) => (
              <div key={idx} className="alert alert-warn">
                {w}
              </div>
            ))}
          </section>

          {/* MODE SWITCHER TAB: Switches Step 3 between Human Solve and Object Match */}
          <div className="mode-tabs" role="tablist" aria-label="解算模式切换">
            <button
              type="button"
              role="tab"
              aria-selected={workMode === "human"}
              className={workMode === "human" ? "active" : ""}
              onClick={() => handleSwitchWorkMode("human")}
            >
              人物解算
              {hasHumanSolve && <IconCheck size={13} className="tab-done" />}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={workMode === "object"}
              className={workMode === "object" ? "active" : ""}
              onClick={() => handleSwitchWorkMode("object")}
            >
              物体匹配
              {hasObjectSolve && <IconCheck size={13} className="tab-done" />}
            </button>
          </div>

          {workMode === "human" ? (
            /* STEP 3 (Human): Unified Target Person Selection + SAM 3D Body Solve */
            <section className="step-card">
              <div className="step-header">
                <div className="step-title">
                  <span className="step-num">3</span>
                  <span>人物三维解算</span>
                </div>
                {hasHumanSolve ? (
                  <span className="tag tag-ok">
                    <IconCheck size={12} />
                    已解算 {shot?.solved_people?.length || 1} 人
                  </span>
                ) : hasDetection || manualEval.people.length > 0 ? (
                  <span className="tag tag-info">
                    选框就绪（{detectedPeople.length} 人{manualEval.people.length > 0 ? ` · 手动 ${manualEval.people.length}` : ""}）
                  </span>
                ) : null}
              </div>

              {/* Sub-block ①: Optional Person Detection & Selection + Manual Keyframe Box */}
              <div className="sub-block">
                <span className="sub-title">
                  ① 人物选择{" "}
                  <span className="muted tnum">
                    {hasDetection || manualEval.people.length > 0
                      ? `（已选 ${selectedIds.size}/${detectedPeople.length} 人）`
                      : ""}
                  </span>
                </span>

                <button
                  type="button"
                  className="btn btn-sm btn-block"
                  disabled={isBusy || !shot?.plate || downloadingFmt !== null}
                  onClick={handleRunDetect}
                >
                  {autoDetectedPeople.length > 0 ? "重新检测人物" : "检测人物"}
                </button>
                {detectionFromOldVersion && (
                  <div className="hint is-warn">
                    人物选框由旧版本生成，解算前将重新检测，人物编号可能变化。
                  </div>
                )}
                {detectionOlderThanGrade && (
                  <div className="hint">
                    调色已在人物检测之后修改，现有选框仍可使用；如需按当前调色重新检测，请点击「重新检测人物」
                    {solvedIdSet.size > 0 ? "（人物编号将重新分配，已解算人物将被清除）" : ""}。
                  </div>
                )}

                {detectedPeople.length > 0 && (
                  <div className={`meta-pill chip-list ${detectedPeople.length > 24 ? "is-scroll" : ""}`}>
                    {detectedPeople.map((p) => {
                      const chosen = selectedIds.has(p.id);
                      return (
                        <button
                          key={p.id}
                          type="button"
                          className={`chip ${chosen ? "is-on" : ""}`}
                          style={{ "--chip": personColor(p.id) } as React.CSSProperties}
                          aria-pressed={chosen}
                          onClick={() => {
                            if (p.is_manual) setActiveManualId(p.id);
                            handleToggleBoxPerson(p.id);
                          }}
                        >
                          {chosen && <IconCheck size={11} />}
                          <span>
                            #{p.id}
                            {p.is_manual ? " 手动" : ""}
                          </span>
                          <span className="chip-note">
                            （{p.frame_count} 帧）{solvedIdSet.has(p.id) ? " · 已解算" : ""}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Manual Keyframe Box Sub-Panel (for individual occluded/missed people) */}
                <div className={`meta-pill is-stack ${manualDrawMode ? "is-active" : ""}`}>
                  <div className="row-between">
                    <span className="sub-title">手动人物框（关键帧插值）</span>
                    <div className="row gap-6 no-shrink">
                      {manualPeople.length > 0 && (
                        <button
                          type="button"
                          data-testid="btn-toggle-manual-draw"
                          aria-pressed={manualDrawMode}
                          className={`btn btn-sm ${manualDrawMode ? "is-on" : ""}`}
                          disabled={isBusy || !shot?.plate}
                          onClick={() => {
                            const nextDraw = !manualDrawMode;
                            setManualDrawMode(nextDraw);
                            if (nextDraw) {
                              setViewMode("camera");
                              setShowBoxes(true);
                              setWipeEnabled(false);
                            }
                          }}
                          title="开启后可在摄影机视角中左键拖拽绘制选框，自动写入关键帧并插值"
                        >
                          {manualDrawMode ? "绘制中" : "绘制选框"}
                        </button>
                      )}
                      <button
                        type="button"
                        data-testid="btn-add-manual-person"
                        className="btn btn-sm"
                        disabled={isBusy || !shot?.plate}
                        onClick={handleAddManualPerson}
                        title="新增一条手动人物框轨迹；关键帧之间自动插值"
                      >
                        + 添加手动框
                      </button>
                    </div>
                  </div>

                  {manualPeople.length === 0 ? null : (
                    <>
                      <div className="chip-list">
                        {manualPeople.map((mp) => {
                          const isAct = activeManualTrack?.id === mp.id;
                          const evP = manualEval.people.find((p) => p.id === mp.id);
                          return (
                            <div key={mp.id} className="chip-group">
                              <button
                                type="button"
                                className={`chip ${isAct ? "is-on" : "is-off"}`}
                                style={isAct ? ({ "--chip": personColor(mp.id) } as React.CSSProperties) : undefined}
                                aria-pressed={isAct}
                                onClick={() => {
                                  setActiveManualId(mp.id);
                                  setShowBoxes(true);
                                }}
                              >
                                {isAct && <span className="chip-dot" />}
                                <span>
                                  #{mp.id}（{mp.keyframes.length} 关键帧 · {evP?.frame_count || 0} 帧）
                                </span>
                              </button>
                              <button
                                type="button"
                                className="btn btn-xs btn-icon btn-ghost"
                                style={{ "--btn-h": "21px" } as React.CSSProperties}
                                title={`删除手动人物 #${mp.id}`}
                                aria-label={`删除手动人物 #${mp.id}`}
                                disabled={isBusy}
                                onClick={() => handleRemoveManualPerson(mp.id)}
                              >
                                <IconClose size={11} />
                              </button>
                            </div>
                          );
                        })}
                      </div>

                      {activeManualTrack && (
                        (() => {
                          const curSt = manualEval.statusesByPerson[activeManualTrack.id]?.[frame] || "inactive";
                          const exactKf = activeManualTrack.keyframes.find((k) => k.frame === frame);
                          const stLabel =
                            curSt === "key_active"
                              ? "◆ 有效关键帧"
                              : curSt === "interp_active"
                              ? "↔ 关键帧间插值"
                              : curSt === "hold_active"
                              ? "→ 延续上一关键帧"
                              : curSt === "occluded"
                              ? "完全遮挡（暂停解算）"
                              : curSt === "off"
                              ? "自此帧起失效"
                              : "未开始（在视口中拖拽绘制）";
                          return (
                            <div className="col gap-6">
                              <div className="row-between" style={{ minHeight: 20 }}>
                                <span className="tnum" style={{ fontSize: 11 }}>
                                  #{activeManualTrack.id} 帧 {frame}：<strong>{stLabel}</strong>
                                </span>
                                {exactKf && (
                                  <button
                                    type="button"
                                    data-testid="btn-manual-del-kf"
                                    className="btn btn-xs btn-danger"
                                    disabled={isBusy}
                                    onClick={handleDeleteManualKeyframeAtCurrentFrame}
                                  >
                                    删除关键帧
                                  </button>
                                )}
                              </div>

                              <div className="row gap-6 row-wrap">
                                <button
                                  type="button"
                                  data-testid="btn-manual-occluded"
                                  aria-pressed={exactKf?.status === "occluded"}
                                  className={`btn btn-sm btn-grow ${exactKf?.status === "occluded" ? "is-on" : ""}`}
                                  disabled={isBusy}
                                  title="标记自当前帧起人物被完全遮挡；重新出现后再次绘制选框即可接续同一人物"
                                  onClick={() => handleSetManualStatusAtFrame("occluded")}
                                >
                                  标记为遮挡
                                </button>
                                <button
                                  type="button"
                                  data-testid="btn-manual-off"
                                  aria-pressed={exactKf?.status === "off"}
                                  className={`btn btn-sm btn-grow ${exactKf?.status === "off" ? "is-on" : ""}`}
                                  disabled={isBusy}
                                  title="标记自当前帧起该人物框失效（人物出画或无需解算）"
                                  onClick={() => handleSetManualStatusAtFrame("off")}
                                >
                                  标记为失效
                                </button>
                                {(curSt === "interp_active" || curSt === "hold_active") && (
                                  <button
                                    type="button"
                                    data-testid="btn-manual-promote-kf"
                                    className="btn btn-sm"
                                    disabled={isBusy}
                                    title="将当前插值框固定为显式关键帧"
                                    onClick={handlePromoteInterpolatedToKeyframe}
                                  >
                                    设为关键帧
                                  </button>
                                )}
                              </div>

                              {activeManualTrack.keyframes.length > 0 && (
                                <div className="row gap-6">
                                  <button
                                    type="button"
                                    data-testid="btn-manual-prev-kf"
                                    className="btn btn-sm btn-grow"
                                    onClick={() => handleJumpManualKeyframe(-1)}
                                  >
                                    <IconChevronLeft size={12} className="icon-lead" />
                                    上一关键帧
                                  </button>
                                  <button
                                    type="button"
                                    data-testid="btn-manual-next-kf"
                                    className="btn btn-sm btn-grow"
                                    onClick={() => handleJumpManualKeyframe(1)}
                                  >
                                    下一关键帧
                                    <IconChevronRight size={12} className="icon-trail" />
                                  </button>
                                </div>
                              )}
                            </div>
                          );
                        })()
                      )}
                    </>
                  )}
                </div>

                <div className="segmented">
                  {(
                    [
                      ["largest", "最大"],
                      ["top", "前 N"],
                      ["all", "全部"],
                      ["ids", "编号"],
                      ["picked", "点选"],
                    ] as const
                  ).map(([k, label]) => (
                    <button
                      key={k}
                      type="button"
                      className={selection.mode === k ? "active" : ""}
                      onClick={() => updateSelection({ ...selection, mode: k })}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {selection.mode === "top" && (
                  <div className="form-row">
                    <span className="form-label">人数 N</span>
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={selection.count}
                      onChange={(e) => updateSelection({ ...selection, count: Number(e.target.value) })}
                      style={{ width: 88 }}
                    />
                  </div>
                )}
                {selection.mode === "ids" && (
                  <div className="form-row">
                    <span className="form-label">人物编号</span>
                    <input
                      type="text"
                      placeholder="如 1,2"
                      value={selection.ids}
                      onChange={(e) => updateSelection({ ...selection, ids: e.target.value })}
                      style={{ flex: 1, minWidth: 0 }}
                    />
                  </div>
                )}
                {selection.mode === "picked" && (
                  <div className="meta-pill is-row">
                    <span>在视口中左键点击人物以添加选点（已添加 {selection.picks.length} 处）</span>
                    {selection.picks.length > 0 && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => updateSelection({ ...selection, picks: [] })}
                      >
                        清除
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* Sub-block ②: SAM 3D Body Solve */}
              <div className="sub-block">
                <span className="sub-title">② 三维解算（SAM 3D Body）</span>

                <div className="row row-wrap gap-12">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={handRefine}
                      onChange={(e) => {
                        const v = e.target.checked;
                        setHandRefine(v);
                        scheduleSavePatch({
                          solve_params: { hand_refine: v, use_mask: useMask, lock_shape: lockShape, smoothing },
                        });
                      }}
                    />
                    手部细节
                  </label>
                </div>
                {handRefine && newSelectedCount >= 8 && (
                  <div className="hint">
                    待解算 {newSelectedCount} 人。无需手指动作时可关闭「手部细节」：逐帧估计速度提高 3 倍以上，身体精度略有下降，手指保持默认姿势。
                  </div>
                )}

                <div className="slider-row" title="整段解算的平滑强度，默认 0.50；数值越高越平稳，越低越贴合逐帧画面">
                  <label>平滑</label>
                  <Range
                    aria-label="平滑"
                    min={0}
                    max={1}
                    step={0.05}
                    value={smoothing}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setSmoothing(v);
                      scheduleSavePatch({
                        solve_params: { hand_refine: handRefine, use_mask: useMask, lock_shape: lockShape, smoothing: v },
                      });
                    }}
                  />
                  <span className="slider-val">{smoothing.toFixed(2)}</span>
                </div>

                {(shot?.solved_people || []).length > 0 && (
                  <div
                    className="qc-panel"
                    title="画面贴合：解算结果与逐帧画面依据之间的像素偏差（中位数）。触地偏移：触地足部在世界空间中相对落点的距离（95% 的帧不超过该值）。体型：相对模型估计身高的整体缩放系数。尺度来源：体型缩放依据扫描地面、摄影机运动或模型估计确定；百分数为测量值所占权重。隐藏帧：未参与解算并以隐藏状态导出的帧数。"
                  >
                    <div className="qc-head">
                      <span>已解算人物（{(shot?.solved_people || []).length}）· 质检读数</span>
                      <button
                        type="button"
                        className="btn btn-xs btn-danger"
                        disabled={isBusy || downloadingFmt !== null}
                        onClick={() => handleRemoveSolvedPeople(null)}
                      >
                        清除
                      </button>
                    </div>
                    <div className="qc-scroll">
                      <table className="qc-table">
                        <thead>
                          <tr>
                            {["人物", "画面贴合", "触地偏移", "体型", "尺度来源", "隐藏帧", ""].map((h, i) => (
                              <th key={`h${i}`} scope="col">
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(shot?.solved_people || []).map((sp) => (
                            <tr key={sp.id}>
                              <td className="qc-id" style={{ color: personColor(sp.id) }}>
                                #{sp.id}
                              </td>
                              <td>{sp.qc ? `${sp.qc.fit_px.toFixed(1)} 像素` : "—"}</td>
                              <td>{sp.qc && sp.qc.foot_hold_cm >= 0 ? `${sp.qc.foot_hold_cm.toFixed(1)} 厘米` : "—"}</td>
                              <td>{sp.qc ? `×${sp.qc.body_scale.toFixed(3)}` : "—"}</td>
                              <td>
                                {!sp.qc
                                  ? "—"
                                  : sp.qc.scale_source.startsWith("ground")
                                    ? "扫描"
                                    : sp.qc.scale_source.startsWith("parallax")
                                      ? "摄影机运动"
                                      : "模型"}
                                {sp.qc && (sp.qc.scale_weight ?? 0) > 0 ? ` ${Math.round((sp.qc.scale_weight ?? 0) * 100)}%` : ""}
                              </td>
                              <td>{sp.qc ? `${sp.qc.hidden_frames} 帧` : "—"}</td>
                              <td>
                                <button
                                  type="button"
                                  className="btn btn-xs btn-icon btn-ghost btn-danger-hover"
                                  title={`移除 #${sp.id} 的解算结果`}
                                  aria-label={`移除 #${sp.id} 的解算结果`}
                                  disabled={isBusy || downloadingFmt !== null}
                                  onClick={() => handleRemoveSolvedPeople([sp.id])}
                                >
                                  <IconClose size={12} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="qc-foot">解算结果为累加：新增解算不影响已解算人物。</div>
                  </div>
                )}

                {(shot?.solve_warnings || []).length > 0 && (
                  <div className="alert alert-warn alert-list">
                    <strong>解算提示</strong>
                    {(shot?.solve_warnings || []).map((w, i) => (
                      <span key={i}>{w}</span>
                    ))}
                  </div>
                )}

                {!shot?.camera && (
                  <div className="alert alert-warn">
                    <div className="alert-row" style={{ alignItems: "center" }}>
                      <span>请先完成第 2 步：导入跟踪摄影机</span>
                      <button type="button" className="btn btn-xs" onClick={focusStep2}>
                        前往导入
                      </button>
                    </div>
                  </div>
                )}

                {isBusy ? (
                  <div className="btn-row">
                    <button type="button" className="btn btn-lg btn-grow is-busy" disabled>
                      <span className="spinner" />
                      {isQueued
                        ? `排队中${shot?.queue_job?.queue_position ? ` (#${shot.queue_job.queue_position})` : ""}`
                        : `GPU ${shot?.queue_job?.gpu_index ?? ""} 计算中...`}
                    </button>
                    <button type="button" className="btn btn-lg btn-danger" onClick={handleCancelTask}>
                      取消
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className={`btn btn-lg btn-block btn-wrap ${
                      humanSolveActionState === "fast_postprocess"
                        ? "btn-fast"
                        : humanSolveActionState === "up_to_date"
                        ? ""
                        : "btn-primary"
                    }`}
                    disabled={
                      !shot?.plate ||
                      !shot?.camera ||
                      downloadingFmt !== null ||
                      (humanSolveActionState === "solve_selected" && selectedIds.size === 0)
                    }
                    title={downloadingFmt !== null ? "导出文件生成中，完成后方可解算" : undefined}
                    onClick={handleRunSolve}
                  >
                    {humanSolveActionState === "up_to_date" && <IconCheck size={13} className="tab-done" />}
                    {humanSolveActionState === "fast_postprocess"
                      ? "按当前平滑重新解算"
                      : humanSolveActionState === "add_people"
                      ? `解算新增的 ${newSelectedCount} 人（保留已解算的 ${solvedIdSet.size} 人）`
                      : humanSolveActionState === "add_people_and_redo"
                      ? `解算新增的 ${newSelectedCount} 人，并按当前设置重新解算已解算的 ${solvedIdSet.size} 人`
                      : humanSolveActionState === "inputs_changed"
                      ? "按当前设置重新解算"
                      : humanSolveActionState === "solve_selected"
                      ? selectedIds.size > 0
                        ? `解算所选的 ${selectedIds.size} 人`
                        : "请先选择待解算人物"
                      : humanSolveActionState === "up_to_date"
                      ? `已完成解算（${solvedIdSet.size} 人）`
                      : "检测并解算所选人物"}
                  </button>
                )}
              </div>
            </section>
          ) : (
            /* STEP 3 (Object): Unified Interactive Click Tracking + 3D Object Reconstruction & 6DoF Pose */
            <section className="step-card">
              <div className="step-header">
                <div className="step-title">
                  <span className="step-num">3</span>
                  <span>物体跟踪与三维解算</span>
                </div>
                <div className="step-actions">
                  {hasObjectSolve ? (
                    <span className="tag tag-ok">
                      <IconCheck size={12} />
                      已完成
                    </span>
                  ) : trackedObjects.length > 0 ? (
                    <span className="tag tag-info">已跟踪 {trackedObjects.length} 个物体</span>
                  ) : null}
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={isBusy}
                    onClick={handleAddNewObject}
                  >
                    + 添加物体
                  </button>
                </div>
              </div>

              {/* Sub-block ①: SAM 3.1 Click & Track */}
              <div className="sub-block">
                <span className="sub-title">① 选择目标物体</span>

                <div className="segmented">
                  <button
                    type="button"
                    className={objectClickLabel === 1 ? "active" : ""}
                    onClick={() => setObjectClickLabel(1)}
                  >
                    + 包含点
                  </button>
                  <button
                    type="button"
                    className={objectClickLabel === 0 ? "active" : ""}
                    onClick={() => setObjectClickLabel(0)}
                  >
                    − 排除点 (Shift+左键)
                  </button>
                </div>

                <div className="meta-pill is-stack">
                  <div className="chip-list">
                    {objectPrompts.map((op) => {
                      const isAct = op.id === activeObjectId;
                      const trInfo = trackedObjects.find((t) => t.id === op.id);
                      const posCnt = (op.points || []).filter((p) => p.label === 1).length;
                      const negCnt = (op.points || []).filter((p) => p.label === 0).length;
                      return (
                        <button
                          key={op.id}
                          type="button"
                          className={`chip ${isAct ? "is-on" : "is-off"}`}
                          style={isAct ? ({ "--chip": objectColor(op.id) } as React.CSSProperties) : undefined}
                          aria-pressed={isAct}
                          onClick={() => setActiveObjectId(op.id)}
                        >
                          {isAct && <span className="chip-dot" />}
                          <span>
                            物体 #{op.id} (+{posCnt}
                            {negCnt > 0 ? `/-${negCnt}` : ""}
                            {trInfo ? ` · ${trInfo.frame_count}帧` : ""})
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  <div className="row-between">
                    <span>在摄影机视角中左键点击目标物体以添加选点</span>
                    <div className="row gap-6 no-shrink">
                      <button
                        type="button"
                        className="btn btn-sm"
                        disabled={isBusy}
                        onClick={() => handleClearObjectPoints(activeObjectId)}
                      >
                        清除选点
                      </button>
                      {objectPrompts.length > 1 && (
                        <button
                          type="button"
                          className="btn btn-sm btn-danger"
                          disabled={isBusy}
                          onClick={() => handleRemoveObjectPrompt(activeObjectId)}
                        >
                          删除
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  className="btn btn-sm btn-block"
                  disabled={isBusy || !shot?.plate || !hasPositiveObjectPoint}
                  onClick={handleRunObjectTrack}
                >
                  {trackedObjects.length > 0 ? "重新预览跟踪掩码" : "预览跟踪掩码（可选）"}
                </button>
              </div>

              {/* Sub-block ②: 3D Object & 6DoF Pose Solve */}
              <div className="sub-block">
                <span className="sub-title">② 物体三维与位姿解算</span>

                <div className="slider-row">
                  <label>平滑</label>
                  <Range
                    aria-label="平滑"
                    min={0}
                    max={1}
                    step={0.05}
                    value={objectSmoothing}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setObjectSmoothing(v);
                      scheduleSavePatch({ object_solve_params: { smoothing: v } });
                    }}
                  />
                  <span className="slider-val">{objectSmoothing.toFixed(2)}</span>
                </div>

                {!shot?.camera && (
                  <div className="alert alert-warn">
                    <div className="alert-row" style={{ alignItems: "center" }}>
                      <span>请先完成第 2 步：导入跟踪摄影机</span>
                      <button type="button" className="btn btn-xs" onClick={focusStep2}>
                        前往导入
                      </button>
                    </div>
                  </div>
                )}

                {isBusy ? (
                  <div className="btn-row">
                    <button type="button" className="btn btn-lg btn-grow is-busy" disabled>
                      <span className="spinner" />
                      {isQueued
                        ? `排队中${shot?.queue_job?.queue_position ? ` (#${shot.queue_job.queue_position})` : ""}`
                        : `GPU ${shot?.queue_job?.gpu_index ?? ""} 计算中...`}
                    </button>
                    <button type="button" className="btn btn-lg btn-danger" onClick={handleCancelTask}>
                      取消
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="btn btn-lg btn-block btn-wrap btn-primary"
                    disabled={!shot?.plate || !shot?.camera || (!hasPositiveObjectPoint && trackedObjects.length === 0)}
                    onClick={handleRunObjectSolve}
                  >
                    {trackedObjects.length > 0
                      ? "解算物体三维与位姿"
                      : "跟踪并解算物体三维与位姿"}
                  </button>
                )}
              </div>
            </section>
          )}

          {/* STEP 4: optional scene scan. Importing it only shows it; "按扫描对齐" puts the solved people on it,
              and "对齐前 / 对齐后" switches between the two kept results. */}
          <section
            className={`step-card ${dragHoverTarget === "step4-scan-card" ? "step-card-drop-active" : ""}`}
            {...makeDropProps(
              "step4-scan-card",
              (files) => handleSceneScanFileUpload(files[0]),
              isBusy || !!uploadingScan || updatingScanOpts
            )}
          >
            <div className="step-header">
              <div className="step-title">
                <span className="step-num">4</span>
                <span>场景扫描</span>
                <span className={`tag ${shot?.scene_scan ? "tag-ok" : "tag-muted"}`}>
                  {shot?.scene_scan ? (
                    <>
                      <IconCheck size={12} />
                      已导入
                    </>
                  ) : (
                    "可选"
                  )}
                </span>
              </div>
              <div className="step-actions">
                <input
                  ref={scanInputRef}
                  type="file"
                  accept=".abc,.fbx,.usd,.usda,.usdc,.usdz"
                  style={{ display: "none" }}
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    e.target.value = "";
                    await handleSceneScanFileUpload(f);
                  }}
                />
                <button
                  type="button"
                  className={`btn btn-sm btn-drop-target ${
                    dragHoverTarget === "btn-scan-upload" ? "btn-drop-active" : ""
                  }`}
                  disabled={isBusy || !!uploadingScan || updatingScanOpts}
                  title="选择场景扫描文件（.abc / .fbx / .usd），或将文件拖拽至此"
                  onClick={() => scanInputRef.current?.click()}
                  {...makeDropProps(
                    "btn-scan-upload",
                    (files) => handleSceneScanFileUpload(files[0]),
                    isBusy || !!uploadingScan || updatingScanOpts
                  )}
                >
                  {uploadingScan
                    ? "正在解析扫描..."
                    : updatingScanOpts
                    ? "正在校准..."
                    : shot?.scene_scan
                    ? "更换扫描"
                    : "导入场景扫描"}
                </button>
                {shot?.scene_scan && (
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    disabled={isBusy || !!uploadingScan || updatingScanOpts}
                    onClick={async () => {
                      if (!shot) return;
                      setScanError(null);
                      const m = await api.clearSceneScan(shot.id);
                      syncLocalFromShot(m);
                      await refreshShotsList();
                      await loadSceneIfReady(m);
                    }}
                  >
                    移除
                  </button>
                )}
              </div>
            </div>

            {scanError && (
              <div className="alert alert-error">
                <div className="alert-row">
                  <span>{scanError}</span>
                  <button
                    type="button"
                    className="btn btn-xs"
                    onClick={() => {
                      setScanError(null);
                      setUiNotice(null);
                    }}
                  >
                    关闭
                  </button>
                </div>
              </div>
            )}

            {(uploadingScan || updatingScanOpts || switchingScanAlign) && (
              <div className="meta-pill is-active">
                <div className="progress-head">
                  <span>
                    {uploadingScan
                      ? uploadingScan.phase === "uploading"
                        ? `正在上传场景扫描文件：${uploadingScan.name}`
                        : `正在解析扫描网格：${uploadingScan.name}...`
                      : switchingScanAlign
                      ? "正在切换对齐状态..."
                      : "正在更新扫描层级物体与单位..."}
                  </span>
                  {uploadingScan && uploadingScan.phase === "uploading" && (
                    <strong className="tnum">{uploadingScan.pct}%</strong>
                  )}
                </div>
                <div className="progress">
                  {uploadingScan && uploadingScan.phase === "uploading" ? (
                    <div className="progress-fill" style={{ width: `${uploadingScan.pct}%` }} />
                  ) : (
                    <div className="progress-indeterminate" />
                  )}
                </div>
              </div>
            )}

            {shot?.scene_scan ? (
              <div className="meta-pill is-stack">
                <div>
                  <strong>{shot.scene_scan.summary.name}</strong> ({shot.scene_scan.summary.format.toUpperCase()}) ·{" "}
                  {shot.scene_scan.summary.raw_face_count.toLocaleString()} 面 · 尺寸{" "}
                  {(shot.scene_scan.summary.span_cm[0] / 100).toFixed(1)}×
                  {(shot.scene_scan.summary.span_cm[1] / 100).toFixed(1)}×
                  {(shot.scene_scan.summary.span_cm[2] / 100).toFixed(1)} m · 用于触地约束与体型尺度
                </div>
                <div className="form-row">
                  <span className="form-label">扫描单位</span>
                  <select
                    className="form-select"
                    disabled={updatingScanOpts || !!uploadingScan}
                    value={shot.scene_scan.unit || "m"}
                    onChange={(e) => handleUpdateSceneScanOpts({ unit: e.target.value })}
                  >
                    <option value="m">米 (m) · 默认</option>
                    <option value="cm">厘米 (cm)</option>
                    <option value="auto">按文件头元数据</option>
                  </select>
                </div>
                {(shot.scene_scan.listing?.meshes?.length || 0) >= 1 && (
                  <div className="form-row">
                    <span className="form-label">扫描层级</span>
                    <button
                      type="button"
                      className={`btn btn-sm btn-split ${
                        (shot.scene_scan.listing?.meshes?.length || 0) > 1 ? "is-on" : ""
                      }`}
                      disabled={updatingScanOpts || !!uploadingScan}
                      onClick={() => setShowScanTreeModal(true)}
                    >
                      <span className="btn-split-value">
                        <span>
                          已选 {shot.scene_scan.summary.mesh_count} /{" "}
                          {shot.scene_scan.summary.total_mesh_count ??
                            shot.scene_scan.listing?.meshes?.length ??
                            shot.scene_scan.summary.mesh_count}{" "}
                          个物体
                        </span>
                      </span>
                      <span className="btn-split-more">
                        选择层级物体
                        <IconChevronDown size={12} className="icon-trail" />
                      </span>
                    </button>
                  </div>
                )}
                {!shot.scan_align_computed && (
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    disabled={isBusy || !!uploadingScan || updatingScanOpts || switchingScanAlign}
                    title="按扫描重新解算已解算人物（使用缓存，不重新估计）"
                    onClick={() => handleScanAlign(true)}
                  >
                    {shot.scan_align ? "重新对齐" : "按扫描对齐"}
                  </button>
                )}
                {(shot.scan_align || shot.scan_align_computed) && (
                  <div className="form-row">
                    <span className="form-label">显示结果</span>
                    <div className="segmented grow">
                      <button
                        type="button"
                        aria-pressed={!shot.scan_align}
                        className={!shot.scan_align ? "active" : ""}
                        disabled={isBusy || !!uploadingScan || updatingScanOpts || switchingScanAlign}
                        onClick={() => {
                          if (shot.scan_align) handleScanAlign(false);
                        }}
                      >
                        对齐前
                      </button>
                      <button
                        type="button"
                        aria-pressed={!!shot.scan_align}
                        className={shot.scan_align ? "active" : ""}
                        disabled={isBusy || !!uploadingScan || updatingScanOpts || switchingScanAlign}
                        onClick={() => {
                          if (!shot.scan_align) handleScanAlign(true);
                        }}
                      >
                        对齐后
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : !uploadingScan ? (
              <div className="hint">导入扫描后，点击「按扫描对齐」进行解算；对齐前、对齐后的结果可切换查看。</div>
            ) : null}
          </section>

          {/* STEP 5: export - each file is built when its download button is pressed */}
          <section className="step-card">
            <div className="step-header">
              <div className="step-title">
                <span className="step-num">5</span>
                <span>导出</span>
              </div>
              <div className="step-actions">
                <div className="segmented auto-width is-small" title="导出文件长度单位（默认米 m）">
                  <button
                    type="button"
                    aria-pressed={exportUnit === "m"}
                    className={exportUnit === "m" ? "active" : ""}
                    onClick={() => handleChangeExportUnit("m")}
                  >
                    米 (m)
                  </button>
                  <button
                    type="button"
                    aria-pressed={exportUnit === "cm"}
                    className={exportUnit === "cm" ? "active" : ""}
                    onClick={() => handleChangeExportUnit("cm")}
                  >
                    厘米 (cm)
                  </button>
                </div>
                {canExport ? (
                  <span className="tag tag-ok">
                    <IconCheck size={12} />
                    可导出
                  </span>
                ) : (
                  <span className="tag tag-muted">待解算</span>
                )}
              </div>
            </div>

            {exportError && (
              <div className="alert alert-error">
                <div className="alert-row">
                  <span>{exportError}</span>
                  <button
                    type="button"
                    className="btn btn-xs"
                    onClick={() => {
                      setExportError(null);
                      setUiNotice(null);
                    }}
                  >
                    关闭
                  </button>
                </div>
              </div>
            )}

            {canExport && solvedIdList.length > 1 && (
              <div className="meta-pill is-stack">
                <div className="row-between">
                  <span className="tnum">
                    导出人物（已选 {exportIdSet.size} / {solvedIdList.length}）
                  </span>
                  <span className="row gap-4 no-shrink">
                    <button
                      type="button"
                      className="btn btn-xs"
                      disabled={exportIdSet.size === solvedIdList.length}
                      onClick={() => saveExportPeople(solvedIdList)}
                    >
                      全选
                    </button>
                    <button
                      type="button"
                      className="btn btn-xs"
                      disabled={exportIdSet.size === solvedIdList.length}
                      onClick={() => saveExportPeople(solvedIdList.filter((id) => !exportIdSet.has(id)))}
                    >
                      反选
                    </button>
                  </span>
                </div>
                <div className={`chip-list ${solvedIdList.length > 40 ? "is-scroll" : ""}`}>
                  {(shot?.solved_people || []).map((sp) => {
                    const chosen = exportIdSet.has(sp.id);
                    return (
                      <button
                        key={sp.id}
                        type="button"
                        className={`chip ${chosen ? "is-on" : "is-off"}`}
                        style={chosen ? ({ "--chip": personColor(sp.id) } as React.CSSProperties) : undefined}
                        aria-pressed={chosen}
                        onClick={() => handleToggleExportPerson(sp.id)}
                      >
                        {chosen && <IconCheck size={11} />}
                        <span>#{sp.id}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="hint">可在视图中点击人物进行选择；未选人物以灰色显示。</div>
              </div>
            )}

            <div className="export-grid">
              {(
                [
                  ["usd_skel", "下载 USD (骨架)"],
                  ["usd_cache", "下载 USD (点缓存)"],
                  ["fbx", "下载 FBX (.fbx)"],
                  ["abc", "下载 ABC (.abc)"],
                ] as const
              ).map(([fmt, label]) => (
                <button
                  key={fmt}
                  type="button"
                  className={`btn ${downloadingFmt === fmt ? "is-busy" : ""}`}
                  disabled={!canExport || isBusy || downloadingFmt !== null}
                  onClick={() => handleDirectDownload(fmt)}
                >
                  {downloadingFmt === fmt && <span className="spinner" />}
                  {downloadingFmt === fmt ? `生成中… ${buildSeconds} 秒` : label}
                </button>
              ))}
            </div>
            {canExport && (
              <div className="hint">文件在下载时生成；点缓存格式（USD 点缓存、ABC）生成时间较长。</div>
            )}
          </section>
        </aside>

        {/* Right 3D Stage & Timeline */}
        <main className="stage-column">
          <div className="viewport-toolbar">
            <div className="toolbar-group">
              <div className="segmented auto-width">
                <button
                  type="button"
                  className={viewMode === "camera" ? "active" : ""}
                  onClick={() => setViewMode("camera")}
                >
                  摄影机视角
                </button>
                <button
                  type="button"
                  className={viewMode === "persp" ? "active" : ""}
                  onClick={() => setViewMode("persp")}
                >
                  透视图
                </button>
              </div>

              <div className="segmented auto-width">
                {workMode === "object" ? (
                  <>
                    <button
                      type="button"
                      className={displayMode === "mesh" ? "active" : ""}
                      onClick={() => setDisplayMode("mesh")}
                    >
                      实体模型
                    </button>
                    <button
                      type="button"
                      className={displayMode === "both" ? "active" : ""}
                      onClick={() => setDisplayMode("both")}
                    >
                      实体+线框
                    </button>
                    <button
                      type="button"
                      className={displayMode === "skeleton" ? "active" : ""}
                      onClick={() => setDisplayMode("skeleton")}
                    >
                      仅线框
                    </button>
                    <button
                      type="button"
                      className={displayMode === "off" ? "active" : ""}
                      onClick={() => setDisplayMode("off")}
                    >
                      关闭模型
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className={displayMode === "both" ? "active" : ""}
                      onClick={() => setDisplayMode("both")}
                    >
                      蒙皮+骨骼
                    </button>
                    <button
                      type="button"
                      className={displayMode === "mesh" ? "active" : ""}
                      onClick={() => setDisplayMode("mesh")}
                    >
                      仅蒙皮
                    </button>
                    <button
                      type="button"
                      className={displayMode === "skeleton" ? "active" : ""}
                      onClick={() => setDisplayMode("skeleton")}
                    >
                      仅骨骼
                    </button>
                    <button
                      type="button"
                      className={displayMode === "off" ? "active" : ""}
                      onClick={() => setDisplayMode("off")}
                    >
                      关闭模型
                    </button>
                  </>
                )}
              </div>

              <button
                type="button"
                aria-pressed={wipeEnabled}
                className={`btn ${wipeEnabled ? "is-on" : ""}`}
                onClick={() => {
                  setWipeEnabled((prev) => {
                    const next = !prev;
                    if (next && displayMode === "off") {
                      setDisplayMode(workMode === "object" ? "mesh" : "both");
                    }
                    return next;
                  });
                }}
                title="拖动滑杆进行左右对比：最左仅显示原片，向右逐步显示模型"
              >
                滑动对比
              </button>
            </div>

            <div className="toolbar-group is-options">
              <label className="toolbar-slider">
                <span>透明度</span>
                <Range
                  min={0.15}
                  max={1.0}
                  step={0.05}
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={autoHide}
                  onChange={(e) => setAutoHide(e.target.checked)}
                />
                自动隐藏
              </label>
              {workMode === "object" ? (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={showObjectMasks}
                    onChange={(e) => setShowObjectMasks(e.target.checked)}
                  />
                  物体掩码
                </label>
              ) : (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={showBoxes}
                    onChange={(e) => setShowBoxes(e.target.checked)}
                  />
                  人物选框
                </label>
              )}
              {scene?.scene_scan && (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={showSceneScan}
                    onChange={(e) => setShowSceneScan(e.target.checked)}
                  />
                  场景扫描
                </label>
              )}
              {viewMode === "camera" && (gateZoom !== 1 || gatePan[0] !== 0 || gatePan[1] !== 0) && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => {
                    setGateZoom(1);
                    setGatePan([0, 0]);
                  }}
                >
                  复位画幅 ({Math.round(gateZoom * 100)}%)
                </button>
              )}
            </div>
          </div>

          <Stage3D
            shotId={shot?.id || ""}
            plateRev={plateRev}
            frame={frame}
            plate={shot?.plate || null}
            grade={grade}
            bypassGrade={bypassGrade}
            scene={scene}
            viewMode={viewMode}
            displayMode={displayMode}
            overlayOpacity={overlayOpacity}
            autoHide={autoHide}
            wipeEnabled={wipeEnabled}
            wipeRatio={wipeRatio}
            onWipeRatioChange={setWipeRatio}
            workMode={workMode}
            showBoxes={showBoxes}
            showObjectMasks={showObjectMasks}
            showSceneScan={showSceneScan}
            detectedPeople={detectedPeople}
            selectedPersonIds={selectedIds}
            selection={selection}
            manualDrawMode={manualDrawMode}
            activeManualPersonId={activeManualTrack?.id ?? null}
            manualStatusesByPerson={manualEval.statusesByPerson}
            onDrawManualBox={handleDrawManualBox}
            objectPrompts={objectPrompts}
            activeObjectId={activeObjectId}
            objectClickLabel={objectClickLabel}
            trackedObjects={trackedObjects}
            onPickPoint={(f, x, y) =>
              updateSelection({ ...selection, picks: [...(selection.picks || []), [f, x, y]] })
            }
            onPickObjectPoint={handlePickObjectPoint}
            onToggleBoxPerson={handleToggleBoxPerson}
            exportPersonIds={solvedIdList.length > 1 ? exportIdSet : undefined}
            onToggleExportPerson={solvedIdList.length > 1 ? handleToggleExportPerson : undefined}
            gateZoom={gateZoom}
            gatePan={gatePan}
            onGateChange={(z, p) => {
              setGateZoom(z);
              setGatePan(p);
            }}
          />

          <footer className="timeline-bar">
            {isBusy && (
              <div className="job-line">
                <span className={`tag ${isQueued ? "tag-warn" : "tag-info"}`}>
                  {isQueued ? "排队中" : "计算中"}
                </span>
                <span className="job-text" title={shot?.stage_message || undefined}>
                  {shot?.stage_message || "等待调度..."}
                </span>
                {isRunning && shot && shot.progress.total > 0 && (
                  <>
                    <div className="progress is-thick">
                      <div
                        className="progress-fill"
                        style={{ width: `${Math.min(100, Math.round((shot.progress.done / shot.progress.total) * 100))}%` }}
                      />
                    </div>
                    <span className="job-pct">
                      {Math.min(100, Math.round((shot.progress.done / shot.progress.total) * 100))}%
                    </span>
                  </>
                )}
              </div>
            )}

            <div className="timeline-controls">
              <button
                type="button"
                className="btn btn-icon"
                onClick={() => setFrame(timelineFrames[0])}
                title="首帧"
                aria-label="首帧"
              >
                <IconSkipStart />
              </button>
              <button
                type="button"
                aria-pressed={playing}
                className={`btn ${playing ? "is-on" : ""}`}
                style={{ minWidth: 76 }}
                onClick={() => setPlaying((p) => !p)}
              >
                {playing ? <IconPause size={12} className="icon-lead" /> : <IconPlay size={12} className="icon-lead" />}
                {playing ? "暂停" : "播放"}
              </button>
              <button
                type="button"
                className="btn btn-icon"
                onClick={() => setFrame(timelineFrames[timelineFrames.length - 1])}
                title="末帧"
                aria-label="末帧"
              >
                <IconSkipEnd />
              </button>

              <span className="frame-readout">
                帧 {frame} <span className="dim">/ {timelineFrames[timelineFrames.length - 1]}</span>
              </span>

              <div className="timeline-track">
                <Range
                  aria-label="时间线"
                  min={0}
                  max={Math.max(0, timelineFrames.length - 1)}
                  value={Math.max(0, timelineFrames.indexOf(frame))}
                  onChange={(e) => {
                    setPlaying(false);
                    setFrame(timelineFrames[Number(e.target.value)] ?? timelineFrames[0]);
                  }}
                />
                {workMode === "human" && activeManualTrack && (
                  <div className="manual-kf-strip" data-testid="manual-keyframe-timeline">
                    <div className="manual-kf-ribbon">
                      {timelineFrames.map((f) => {
                        const st = manualEval.statusesByPerson[activeManualTrack.id]?.[f] || "inactive";
                        const bg =
                          st === "key_active" || st === "interp_active" || st === "hold_active"
                            ? "rgba(79, 141, 255, 0.85)"
                            : st === "occluded"
                            ? "rgba(231, 162, 61, 0.8)"
                            : st === "off"
                            ? "rgba(234, 92, 92, 0.5)"
                            : "rgba(43, 52, 66, 0.9)";
                        return <div key={f} style={{ flex: 1, background: bg }} />;
                      })}
                    </div>
                    {activeManualTrack.keyframes.length > 0 && (
                      <div className="manual-kf-pins">
                        {activeManualTrack.keyframes.map((kf) => {
                          const idx = timelineFrames.indexOf(kf.frame);
                          if (idx < 0) return null;
                          const pct =
                            timelineFrames.length <= 1
                              ? 0
                              : (idx / Math.max(1, timelineFrames.length - 1)) * 100;
                          const isCur = kf.frame === frame;
                          const sym =
                            kf.status === "active" ? "◆" : kf.status === "occluded" ? "◇" : "✕";
                          const col =
                            kf.status === "active"
                              ? "#8db5ff"
                              : kf.status === "occluded"
                              ? "#f1c06c"
                              : "#f69b9b";
                          const label =
                            kf.status === "active"
                              ? "有效关键帧"
                              : kf.status === "occluded"
                              ? "完全遮挡"
                              : "自此帧起失效";
                          return (
                            <button
                              key={kf.frame}
                              type="button"
                              data-testid={`manual-kf-pin-${kf.frame}`}
                              className={`manual-kf-pin ${isCur ? "is-current" : ""}`}
                              style={{ left: `clamp(7px, ${pct}%, calc(100% - 7px))`, borderColor: col, color: col }}
                              title={`手动 #${activeManualTrack.id} · 帧 ${kf.frame} · ${label}（点击跳转）`}
                              onClick={() => {
                                setPlaying(false);
                                setFrame(kf.frame);
                              }}
                            >
                              {sym}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
                {scene && (scene.characters.length > 0 || (scene.objects?.length || 0) > 0) && (
                  <div className="solved-ribbon" title="绿色：已解算帧；红色：未解算帧">
                    {timelineFrames.map((f) => (
                      <div key={f} className={solvedFramesSet.has(f) ? "is-solved" : undefined} />
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="qa-strip">
              <div className="qa-left">
                {/* while a job runs its message is on the job line above: it is not said twice */}
                {!isBusy && <span>{shot?.stage_message || "就绪"}</span>}
                {shot?.alignment_report?.warnings?.map((w, idx) => (
                  <span key={idx} className="tag tag-warn is-wrap">
                    {w}
                  </span>
                ))}
              </div>
              <span className="qa-hint">
                {viewMode === "persp"
                  ? solvedIdList.length > 1 && workMode === "human"
                    ? "左键旋转 · 中键平移 · 右键缩放 · F 居中 · 单击人物：切换导出"
                    : "左键旋转 · 中键平移 · 右键缩放 · F 居中"
                  : workMode === "object"
                  ? "左键添加包含点 · Shift+左键添加排除点 · 中键平移 · 右键缩放"
                  : manualDrawMode && activeManualTrack
                  ? `绘制模式（#${activeManualTrack.id}）：左键拖拽绘制选框 · 中键平移 · 右键缩放`
                  : solvedIdList.length > 1 && selection.mode !== "picked"
                  ? "中键平移 · 右键缩放 · 左键点击选框：选择待解算人物 · 点击已解算人物：切换导出"
                  : "中键平移 · 右键缩放 · 左键点击选框：选择待解算人物"}
              </span>
            </div>
          </footer>
        </main>
      </div>

      {showGalleryModal && (
        <TaskGalleryModal
          shots={shots}
          currentShotId={shot?.id || ""}
          onSelectTask={(sid, pid, jump4) => selectShot(sid, pid, jump4)}
          machineLabel={machineLabel}
          onClaimTask={async (sid) => {
            try {
              const m = await api.claimShot(sid);
              await refreshShotsList();
              if (shot?.id === sid) syncLocalFromShot(m);
            } catch (err: any) {
              setUiNotice(err?.message || String(err));
            }
          }}
          onDeleteTask={async (sid) => {
            leavingRef.current = true; // this computer deletes it itself: no notice
            try {
              await api.deleteShot(sid);
              const nextList = await refreshShotsList();
              if (shot?.id === sid) {
                const nextOwn = nextList.find((s) => !s.is_public);
                if (nextOwn) {
                  await selectShot(nextOwn.id);
                } else {
                  const c = await api.createShot("任务 01");
                  await refreshShotsList();
                  syncLocalFromShot(c, true);
                  setScene(null);
                }
              }
            } finally {
              leavingRef.current = false;
            }
          }}
          onOpenCreate={() => setShowNewTaskModal(true)}
          onClose={() => setShowGalleryModal(false)}
        />
      )}

      {showNewTaskModal && (
        <NewTaskModal
          defaultName={`任务 ${String(shots.filter((s) => !s.is_public).length + 1).padStart(2, "0")}`}
          onConfirm={async (taskName) => {
            await flushPendingPatch();
            const created = await api.createShot(taskName);
            await refreshShotsList();
            syncLocalFromShot(created, true);
            setScene(null);
            setShowNewTaskModal(false);
          }}
          onClose={() => setShowNewTaskModal(false)}
        />
      )}

      {showCamTreeModal && shot?.camera && (
        <CameraHierarchyModal
          format={shot.camera.listing.format}
          unitLabel={
            (shot.camera.unit || "m") === "m"
              ? "米 (m)"
              : shot.camera.unit === "cm"
              ? "厘米 (cm)"
              : "按文件头"
          }
          unitToCm={shot.camera.resolved_unit_to_cm ?? 100}
          cameras={shot.camera.listing.cameras}
          hierarchy={shot.camera.listing.hierarchy}
          currentCameraPath={shot.camera.camera_path}
          onConfirm={async (cameraPath) => {
            await handleUpdateCameraOpts({ camera_path: cameraPath });
          }}
          onClose={() => setShowCamTreeModal(false)}
        />
      )}

      {showScanTreeModal && shot?.scene_scan && (shot.scene_scan.listing?.meshes?.length || 0) > 0 && (
        <SceneScanHierarchyModal
          format={shot.scene_scan.summary.format}
          meshes={shot.scene_scan.listing!.meshes}
          hierarchy={shot.scene_scan.listing!.hierarchy}
          selectedPaths={shot.scene_scan.selected_paths}
          onConfirm={async (selectedPaths) => {
            await handleUpdateSceneScanOpts({ selected_paths: selectedPaths });
          }}
          onClose={() => setShowScanTreeModal(false)}
        />
      )}

      <ConfirmHost />
    </div>
  );
};
