import React, { useMemo, useState } from "react";
import { ShotSummary } from "../types";
import { IconCheck } from "./Icons";
import { Modal } from "./Modal";

export const NewTaskModal: React.FC<{
  defaultName: string;
  onConfirm: (name: string) => Promise<void> | void;
  onClose: () => void;
}> = ({ defaultName, onConfirm, onClose }) => {
  const [name, setName] = useState(defaultName);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || submitting) return;
    setSubmitting(true);
    try {
      await onConfirm(trimmed);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="新建任务"
      width={420}
      onClose={onClose}
      onSubmit={handleSubmit}
      footer={
        <div className="modal-footer-actions">
          <button type="button" className="btn" onClick={onClose}>
            取消
          </button>
          <button type="submit" className={`btn btn-primary ${submitting ? "is-busy" : ""}`} disabled={!name.trim() || submitting}>
            {submitting && <span className="spinner" />}
            {submitting ? "创建中..." : "创建任务"}
          </button>
        </div>
      }
    >
      <div className="field">
        <label htmlFor="new-task-name">任务名称</label>
        <input
          id="new-task-name"
          type="text"
          className="is-large"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="任务名称，如 shot_010"
          data-initial-focus
          required
        />
      </div>
    </Modal>
  );
};

export const TaskGalleryModal: React.FC<{
  shots: ShotSummary[];
  currentShotId: string;
  machineLabel?: string;
  onSelectTask: (shotId: string, personId?: number, jumpToStep4?: boolean) => void;
  onDeleteTask: (shotId: string) => Promise<void>;
  onClaimTask: (shotId: string) => Promise<void>;
  onOpenCreate: () => void;
  onClose: () => void;
}> = ({ shots, currentShotId, machineLabel, onSelectTask, onDeleteTask, onClaimTask, onOpenCreate, onClose }) => {
  // v0.2.12: this computer's tasks, and the public ones (tasks from before per-computer tasks whose computer
  // could not be decided - each of them is moved into its computer's list with one click)
  const [filter, setFilter] = useState<"all" | "solved" | "public">("all");
  const [search, setSearch] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [claimingId, setClaimingId] = useState<string | null>(null);

  const mine = useMemo(() => shots.filter((s) => !s.is_public), [shots]);
  const publicTasks = useMemo(() => shots.filter((s) => s.is_public), [shots]);
  const shownFilter = filter === "public" && publicTasks.length === 0 ? "all" : filter;

  const filtered = useMemo(() => {
    const pool = shownFilter === "public" ? publicTasks : mine;
    return pool.filter((s) => {
      if (shownFilter === "solved" && !s.has_solve_cache) return false;
      if (search.trim() && !s.name.toLowerCase().includes(search.trim().toLowerCase())) return false;
      return true;
    });
  }, [mine, publicTasks, shownFilter, search]);

  const solvedCount = useMemo(() => mine.filter((s) => s.has_solve_cache).length, [mine]);

  return (
    <Modal
      title="人物与任务列表"
      width={920}
      maxHeight="86vh"
      onClose={onClose}
      titleExtra={
        <>
          <div className="segmented auto-width">
            <button
              type="button"
              aria-pressed={shownFilter === "all"}
              className={shownFilter === "all" ? "active" : ""}
              onClick={() => setFilter("all")}
            >
              我的任务 ({mine.length})
            </button>
            <button
              type="button"
              aria-pressed={shownFilter === "solved"}
              className={shownFilter === "solved" ? "active" : ""}
              onClick={() => setFilter("solved")}
            >
              已解算 ({solvedCount})
            </button>
            {publicTasks.length > 0 && (
              <button
                type="button"
                aria-pressed={shownFilter === "public"}
                className={shownFilter === "public" ? "active" : ""}
                onClick={() => setFilter("public")}
              >
                未分类任务 ({publicTasks.length})
              </button>
            )}
          </div>
          <input
            type="text"
            placeholder="搜索名称..."
            aria-label="搜索名称"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: 180 }}
          />
        </>
      }
      headerActions={
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => {
            onClose();
            onOpenCreate();
          }}
        >
          + 新建任务
        </button>
      }
    >
      <div className="hint task-gallery-hint">
        {shownFilter === "public"
          ? "升级到按电脑区分任务之前建的、未能确定属于哪台电脑的任务，所有电脑都能看到。是你的任务就点「移到我的任务」，之后只有本机能看到；也可以请管理员在后台分配。"
          : `这里只显示本机${machineLabel ? `（${machineLabel}）` : ""}建的任务，其他电脑看不到。`}
      </div>
      {filtered.length === 0 ? (
        <div className="meta-pill is-empty">暂无匹配项</div>
      ) : (
        <div className="task-gallery-grid">
          {filtered.map((s) => {
            const isActive = s.id === currentShotId;
            const isBusy = s.status === "queued" || s.status === "running";
            return (
              <div
                key={s.id}
                className={`task-card ${isActive ? "task-card-active" : ""}`}
                onMouseLeave={() => {
                  // moving away from the card takes back an unconfirmed "删除"
                  if (confirmDeleteId === s.id && deletingId !== s.id) setConfirmDeleteId(null);
                }}
              >
                {/* Clean 16:9 Thumbnail Cover (no detection boxes) */}
                <div
                  className="task-thumb-wrap"
                  onClick={() => {
                    onSelectTask(s.id, undefined, s.has_solve_cache);
                    onClose();
                  }}
                >
                  {s.has_plate ? (
                    <img
                      src={`/api/shots/${encodeURIComponent(s.id)}/thumbnail`}
                      alt={s.name}
                      className="task-thumb-img"
                      loading="lazy"
                    />
                  ) : (
                    <div className="task-thumb-empty">未上传序列帧</div>
                  )}

                  {s.plate && (
                    <div className="task-thumb-meta">
                      {s.plate.width}×{s.plate.height} · {s.plate.first_frame}–{s.plate.last_frame} (
                      {s.plate.frame_count}帧) · {s.plate.file_ext.toUpperCase()}
                    </div>
                  )}
                </div>

                {/* Clean Task & Solved Character Info Body */}
                <div className="task-card-body">
                  <div className="row-between">
                    <strong className="task-card-name" title={s.name}>
                      {s.name}
                    </strong>

                    {isBusy ? (
                      <span className="tag tag-warn">
                        {s.status === "running"
                          ? `GPU ${s.gpu_index ?? ""} 计算中`
                          : `排队 #${s.queue_position ?? ""}`}
                      </span>
                    ) : s.has_solve_cache ? (
                      <span className="tag tag-ok">
                        <IconCheck size={12} />
                        {s.has_camera ? "已对齐摄影机" : "已解算"}
                      </span>
                    ) : (
                      <span className="tag tag-muted">未解算</span>
                    )}
                  </div>

                  <div className="task-card-facts">
                    <div>
                      {s.people_count > 0 && (s.objects_count || 0) > 0
                        ? `已解算: ${s.people_count} 人 + ${s.objects_count} 物体`
                        : (s.objects_count || 0) > 0
                        ? `物体: 已解算 ${s.objects_count} 个`
                        : s.people_count > 0
                        ? `人物: 已解算 ${s.people_count} 人`
                        : s.detected_count > 0
                        ? `人物: 已检测 ${s.detected_count} 人（待解算）`
                        : (s.tracked_objects_count || 0) > 0
                        ? `物体: 已跟踪 ${s.tracked_objects_count} 个（待解算）`
                        : "状态: 未解算"}
                      {s.plate ? ` · ${s.plate.resolved_colorspace}` : ""}
                    </div>
                    <div>
                      {s.camera ? `摄影机: ${s.camera.name}` : "摄影机: 未导入"}
                      {s.has_scene_scan ? " + 场景扫描" : ""}
                      {s.expires_in_days != null && s.has_solve_cache
                        ? ` · 剩余保留 ${Math.ceil(s.expires_in_days)} 天`
                        : ""}
                    </div>
                  </div>

                  <div className="task-card-actions">
                    <button
                      type="button"
                      className={`btn btn-sm btn-grow ${isActive ? "is-on" : ""}`}
                      onClick={() => {
                        onSelectTask(s.id, undefined, false);
                        onClose();
                      }}
                    >
                      {s.has_solve_cache ? "打开已解算任务" : "打开任务"}
                    </button>

                    {s.is_public ? (
                      <button
                        type="button"
                        className={`btn btn-sm btn-primary ${claimingId === s.id ? "is-busy" : ""}`}
                        disabled={claimingId === s.id}
                        onClick={async () => {
                          setClaimingId(s.id);
                          try {
                            await onClaimTask(s.id);
                          } finally {
                            setClaimingId(null);
                          }
                        }}
                      >
                        {claimingId === s.id && <span className="spinner" />}
                        移到我的任务
                      </button>
                    ) : confirmDeleteId === s.id ? (
                      <button
                        type="button"
                        className={`btn btn-sm btn-danger is-confirm ${deletingId === s.id ? "is-busy" : ""}`}
                        disabled={deletingId === s.id}
                        onClick={async () => {
                          setDeletingId(s.id);
                          try {
                            await onDeleteTask(s.id);
                          } finally {
                            setDeletingId(null);
                            setConfirmDeleteId(null);
                          }
                        }}
                      >
                        {deletingId === s.id && <span className="spinner" />}
                        确认删除
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="btn btn-sm btn-danger"
                        disabled={isBusy}
                        onClick={() => setConfirmDeleteId(s.id)}
                      >
                        删除
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Modal>
  );
};
