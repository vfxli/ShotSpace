export interface GradeParams {
  exposure: number;
  brightness: number;
  contrast?: number;
  saturation?: number;
  gamma: number;
}

export interface OcioSettings {
  input_colorspace: string;
  ocio_view: string;
}

export interface OcioOptionSet {
  config_name: string;
  colorspaces: Array<{ id: string; label: string }>;
  views: Array<{ id: string; label: string }>;
}

export interface PlateMeta {
  source_name: string;
  kind: "sequence";
  width: number;
  height: number;
  fps: number;
  frames: number[];
  first_frame: number;
  last_frame: number;
  frame_count: number;
  file_ext: string;
  bit_depth: string;
  input_colorspace: string;
  resolved_colorspace: string;
  ocio_view: string;
  warnings?: string[];
}

export interface HierarchyCameraInfo {
  name: string;
  path: string;
  frames: number[];
  frame_count?: number;
  focal_mm: number;
  h_aperture_mm?: number;
  travel_raw?: number;
}

export interface HierarchyMeshInfo {
  name: string;
  path: string;
  vertex_count: number;
  face_count: number;
}

export interface HierarchyNode {
  name: string;
  path: string;
  kind: "group" | "camera" | "mesh" | "character";
  camera?: HierarchyCameraInfo | null;
  mesh?: HierarchyMeshInfo | null;
  character?: { name: string; path: string; joints: number } | null;
  camera_count: number;
  mesh_count: number;
  face_count: number;
  children: HierarchyNode[];
}

export interface CameraSummary {
  name: string;
  path: string;
  frames: number[];
  frame_count: number;
  authored_count: number;
  focal_mm: number | [number, number];
  is_zoom: boolean;
  h_aperture_mm: number;
  v_aperture_mm: number;
  center_mm: [number, number];
  pixel_aspect: number;
  width: number;
  height: number;
  fps: number;
  travel_cm?: number;
  span_cm?: [number, number, number];
  source_file: string;
}

export interface CameraEntry {
  file_path: string;
  camera_path: string;
  unit?: string;
  resolved_unit_to_cm?: number;
  fbx_take: string;
  frame_offset: number;
  listing: {
    format: string;
    fps: number;
    unit_to_cm?: number;
    cameras: HierarchyCameraInfo[];
    default_camera_path?: string;
    hierarchy?: HierarchyNode[];
    takes: string[];
    auto_take: string;
  };
  summary: CameraSummary;
}

export interface SceneScanSummary {
  name: string;
  source_file: string;
  format: string;
  unit: string;
  resolved_unit_to_cm: number;
  mesh_count: number;
  total_mesh_count?: number;
  selected_paths?: string[];
  raw_vertex_count: number;
  raw_face_count: number;
  vertex_count: number;
  face_count: number;
  bbox_min_cm: [number, number, number];
  bbox_max_cm: [number, number, number];
  span_cm: [number, number, number];
}

export interface SceneScanEntry {
  file_path: string;
  unit: string;
  selected_paths?: string[];
  listing?: {
    format: string;
    meshes: HierarchyMeshInfo[];
    hierarchy: HierarchyNode[];
  };
  summary: SceneScanSummary;
}

export type ManualKeyframeStatus = "active" | "occluded" | "off";

export type ManualEvaluatedFrameStatus =
  | "inactive"
  | "key_active"
  | "interp_active"
  | "hold_active"
  | "occluded"
  | "off";

export interface ManualBoxKeyframe {
  frame: number;
  status: ManualKeyframeStatus;
  box?: [number, number, number, number];
}

export interface ManualPersonTrack {
  id: number;
  name?: string;
  enabled?: boolean;
  keyframes: ManualBoxKeyframe[];
}

export interface DetectedPerson {
  id: number;
  name?: string;
  is_manual?: boolean;
  prominence: number;
  frame_count: number;
  boxes: Record<string, [number, number, number, number]>;
}

export interface SelectionState {
  mode: "largest" | "top" | "all" | "ids" | "picked";
  count: number;
  ids: string;
  picks: Array<[number, number, number]>;
  least_frames: number;
}

export interface AlignmentReport {
  mode: string;
  fit_requested: string;
  frames: number[];
  common_authored_frames: number;
  scale: number;
  rotation_deg: number;
  translation_cm: number;
  rms_pos_error_cm: number;
  rms_ang_error_deg: number;
  lens_diff_pct: number;
  target_spread_cm: number;
  scale_locked_still: boolean;
  track_reversed: boolean;
  warnings: string[];
}

export interface ExportRecord {
  filename: string;
  format: string;
  unit: string;
  fps: number;
  size_bytes: number;
  created_at: string;
}

export interface QueueJobInfo {
  id: string;
  shot_id: string;
  shot_name: string;
  job_type: "detect" | "solve";
  client_ip: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  gpu_index: number | null;
  stage: string;
  message: string;
  done: number;
  total: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  queue_position?: number;
  duration_sec?: number | null;
  wait_sec?: number;
}

export interface PublicQueueSummary {
  queued_count: number;
  running_count: number;
  enabled_gpu_count: number;
  busy_gpu_count?: number;
  total_gpu_count: number;
  all_gpus_disabled: boolean;
}

export interface PersonInfoSummary {
  id: number;
  frame_count: number;
  first_frame: number;
  last_frame: number;
  solved: boolean;
}

export type WorkMode = "human" | "object";

export interface ObjectClickPoint {
  frame: number;
  x: number;
  y: number;
  label: 0 | 1;
}

export interface ObjectPromptItem {
  id: number;
  name: string;
  points: ObjectClickPoint[];
}

export interface TrackedObjectInfo {
  id: number;
  name: string;
  frame_count: number;
  first_frame: number;
  last_frame: number;
  boxes: Record<string, [number, number, number, number]>;
  contours: Record<string, Array<Array<[number, number]>>>;
  points: ObjectClickPoint[];
}

export interface SolvedObjectSummary {
  id: number;
  name: string;
  file: string;
  frames: number[];
  solved_frames?: number[];
  visible_count?: number;
  vertex_count?: number;
  face_count?: number;
}

export interface ShotMeta {
  id: string;
  name: string;
  work_mode?: WorkMode;
  status: "idle" | "queued" | "running" | "ready" | "error";
  active_job_id: string | null;
  queue_job?: QueueJobInfo | null;
  has_solve_cache?: boolean;
  has_human_solve_cache?: boolean;
  has_object_solve_cache?: boolean;
  space_mode?: "none" | "origin" | "aligned";
  expires_in_days?: number | null;
  retention_days?: number;
  solved_at?: string | null;
  stage_message: string;
  progress: { done: number; total: number; stage: string };
  error: string | null;
  created_at: string;
  updated_at: string;
  plate: PlateMeta | null;
  ocio: OcioSettings;
  grade: GradeParams;
  detect_params: { threshold: number };
  manual_people?: ManualPersonTrack[];
  detection: {
    people: DetectedPerson[];
    grade?: GradeParams;
    threshold: number;
    ocio_cs?: string;
    ocio_view?: string;
    /** which tracker wrote the boxes; boxes of the v0.1.x detector carry no tag */
    engine?: string;
    run_id?: string;
  } | null;
  selection: SelectionState;
  solve_params: { hand_refine: boolean; use_mask?: boolean; lock_shape: boolean; smoothing: number };
  solved_state?: {
    cam_file: string;
    cam_path: string;
    cam_take: string;
    grade: GradeParams;
    ocio_cs: string;
    ocio_view: string;
    threshold: number;
    selected_ids: number[];
    manual_sig?: string;
    manual_tracks?: ManualPersonTrack[];
    hand_refine: boolean;
    use_mask: boolean;
    lock_shape: boolean;
    smoothing: number;
  } | null;
  cache_signatures: Record<string, any>;
  solved_people: Array<{
    id: number;
    file: string;
    frames: number[];
    qc?: {
      fit_px: number;
      fit_px_p90: number;
      foot_hold_cm: number;
      planted_pct: number;
      body_scale: number;
      scale_source: string;
      scale_weight?: number;
      solved_frames: number;
      hidden_frames: number;
    };
  }>;
  solve_warnings?: string[];
  active_person_ids: number[];
  /** Solved people chosen for export; an empty list means everybody. */
  export_person_ids?: number[];
  object_prompts?: ObjectPromptItem[];
  object_tracking?: {
    frames: number[];
    image_size: [number, number];
    objects: TrackedObjectInfo[];
  } | null;
  object_solve_params?: { smoothing: number };
  solved_objects?: SolvedObjectSummary[];
  active_object_ids?: number[];
  camera: CameraEntry | null;
  scene_scan?: SceneScanEntry | null;
  /** step 4: true = the state "对齐后" (the scan is in use) is on show, false = "对齐前" */
  scan_align?: boolean;
  /** step 4: the results of "对齐后" were computed with the scan as it is now */
  scan_align_computed?: boolean;
  /** v0.2.12: an unclassified task - made before tasks belonged to computers, its computer could not be decided:
 * every computer sees it until it is moved into one computer's list (by that computer, or by the admin) */
  is_public?: boolean;
  /** v0.2.12: the computer the task belongs to, "WS-023 (10.0.3.23)" / "公共任务" */
  owner_label?: string;
  alignment_report: AlignmentReport | null;
  has_view: boolean;
  exports: ExportRecord[];
}

export interface ShotSummary {
  id: string;
  name: string;
  work_mode?: WorkMode;
  status: string;
  queue_position: number | null;
  gpu_index: number | null;
  has_plate: boolean;
  has_view: boolean;
  has_solve_cache: boolean;
  has_human_solve_cache?: boolean;
  has_object_solve_cache?: boolean;
  has_camera: boolean;
  has_scene_scan?: boolean;
  detected_count: number;
  people_count: number;
  objects_count?: number;
  tracked_objects_count?: number;
  people_info: PersonInfoSummary[];
  active_person_ids: number[];
  plate: PlateMeta | null;
  camera: CameraSummary | null;
  expires_in_days?: number | null;
  updated_at: string;
  created_at: string;
  is_public?: boolean;
  owner_label?: string;
}

/** v0.2.12: the computer this page runs on, as the server sees it */
export interface MachineInfo {
  ip: string;
  name: string;
}

/** v0.2.12, admin: every task with the computer it belongs to */
export interface AdminTaskOwner {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  is_public: boolean;
  owner_label: string;
  owner_how: string;
  owner_why: string;
  owner_key: string;
  owner_ip: string;
  owner_name: string;
  people_count: number;
  status: string;
}

/** v0.2.12, admin: a computer, how many tasks it has, when it last opened the service; the first entry is "未分类" */
export interface AdminMachine {
  key: string;
  label: string;
  ip: string;
  name: string;
  task_count: number;
  last_seen: string;
  unclassified?: boolean;
}

export interface AdminGpuItem {
  index: number;
  name: string;
  compute_cap: string;
  sm: string;
  compatible: boolean;
  enabled: boolean;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  utilization_pct: number;
  temperature_c: number;
  state: "busy" | "idle" | "disabled";
  current_job: QueueJobInfo | null;
}

export interface AdminOverview {
  summary: PublicQueueSummary;
  gpus: AdminGpuItem[];
  queued: QueueJobInfo[];
  running: QueueJobInfo[];
  history: QueueJobInfo[];
  server?: {
    port: number;
    host: string;
    cache_dir: string;
    configured_cache_dir: string;
    default_cache_dir: string;
    is_custom_cache_dir: boolean;
    disk_free_gb: number;
    disk_total_gb: number;
    retention_days: number;
    total_mb: number;
    shots_total: number;
    solved_shots: number;
    updated_at: string;
    /** data collection: the folder of the daily tables and their size */
    data_collect_dir?: string;
    data_collect_today_mb?: number;
    data_collect_total_mb?: number;
  };
}

export interface BinSlice {
  offset: number;
  byteLength: number;
  length: number;
  dtype: string;
  shape: number[];
}

export interface ViewManifest {
  version: number;
  frames: number[];
  plate: PlateMeta | null;
  cameras: Array<{
    role: string;
    name: string;
    path: string;
    frames: number[];
    width: number;
    height: number;
    pixel_aspect: number;
    overscan: number[];
    fps: number;
    world: BinSlice;
    focal_mm: BinSlice;
    h_aperture_mm: BinSlice;
    v_aperture_mm: BinSlice;
    center_mm: BinSlice;
  }>;
  characters: Array<{
    name: string;
    person_id: number;
    frames: number[];
    visible?: number[];
    joint_names: string[];
    parents: number[];
    vertex_count: number;
    face_count: number;
    rest_points: BinSlice;
    faces: BinSlice;
    skin_indices: BinSlice;
    skin_weights: BinSlice;
    bind_world: BinSlice;
    inv_bind: BinSlice;
    anim_world: BinSlice;
  }>;
  objects?: Array<{
    name: string;
    object_id: number;
    frames: number[];
    visible?: number[];
    vertex_count: number;
    face_count: number;
    rest_points: BinSlice;
    faces: BinSlice;
    vertex_colors: BinSlice | null;
    world: BinSlice;
  }>;
  scene_scan?: {
    name: string;
    vertex_count: number;
    face_count: number;
    vertices: BinSlice;
    faces: BinSlice;
  } | null;
  alignment: AlignmentReport | null;
  binary: { raw_bytes: number; gz_bytes: number };
}

export interface DecodedCamera {
  role: string;
  name: string;
  path: string;
  frames: number[];
  frameToIndex: Map<number, number>;
  width: number;
  height: number;
  pixel_aspect: number;
  overscan: number[];
  fps: number;
  world: Float32Array;
  focal_mm: Float32Array;
  h_aperture_mm: Float32Array;
  v_aperture_mm: Float32Array;
  center_mm: Float32Array;
}

export interface DecodedCharacter {
  name: string;
  person_id: number;
  frames: number[];
  visible: number[];
  frameToIndex: Map<number, number>;
  joint_names: string[];
  parents: number[];
  rest_points: Float32Array;
  faces: Uint32Array;
  skin_indices: Uint16Array;
  skin_weights: Float32Array;
  bind_world: Float32Array;
  inv_bind: Float32Array;
  anim_world: Float32Array;
}

export interface DecodedObject {
  name: string;
  object_id: number;
  frames: number[];
  visible: number[];
  frameToIndex: Map<number, number>;
  rest_points: Float32Array;
  faces: Uint32Array;
  vertex_colors: Float32Array | null;
  world: Float32Array;
}

export interface DecodedSceneScan {
  name: string;
  vertex_count: number;
  face_count: number;
  vertices: Float32Array;
  faces: Uint32Array;
}

export interface DecodedScene {
  manifest: ViewManifest;
  cameras: DecodedCamera[];
  characters: DecodedCharacter[];
  objects: DecodedObject[];
  scene_scan?: DecodedSceneScan | null;
}

export const OBJECT_PALETTE = [
  "#f59e0b",
  "#2dd4bf",
  "#a78bfa",
  "#fb7185",
  "#38bdf8",
  "#34d399",
];

export function objectColor(objectId: number): string {
  return OBJECT_PALETTE[Math.max(0, objectId - 1) % OBJECT_PALETTE.length];
}

export const PERSON_PALETTE = [
  "#38bdf8",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#a78bfa",
  "#fb7185",
  "#2dd4bf",
  "#f97316",
];

export function personColor(personId: number): string {
  return PERSON_PALETTE[Math.max(0, personId - 1) % PERSON_PALETTE.length];
}

/** Tag of the people tracker whose boxes the solver can use (`people_tracker.ENGINE_TAG` on the server). */
export const TRACKER_ENGINE = "sam31_native_people_v1";
