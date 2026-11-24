import React, { useMemo, useState } from "react";
import { HierarchyCameraInfo, HierarchyMeshInfo, HierarchyNode } from "../types";
import { IconCamera, IconChevronDown, IconCube, IconFolder } from "./Icons";
import { Modal } from "./Modal";

function buildFallbackTreeFromCameras(cameras: HierarchyCameraInfo[]): HierarchyNode[] {
  const map = new Map<string, HierarchyNode>();
  const roots: HierarchyNode[] = [];
  for (const c of cameras) {
    const segs = (c.path || `/${c.name}`).split("/").filter(Boolean);
    let curList = roots;
    let prefix = "";
    let last: HierarchyNode | null = null;
    for (const s of segs) {
      prefix = `${prefix}/${s}`;
      let nd = map.get(prefix);
      if (!nd) {
        nd = {
          name: s,
          path: prefix,
          kind: "group",
          camera_count: 0,
          mesh_count: 0,
          face_count: 0,
          children: [],
        };
        map.set(prefix, nd);
        curList.push(nd);
      }
      curList = nd.children;
      last = nd;
    }
    if (last) {
      last.kind = "camera";
      last.camera = c;
    }
  }
  const rollup = (n: HierarchyNode): number => {
    let tot = n.camera ? 1 : 0;
    for (const ch of n.children) tot += rollup(ch);
    n.camera_count = tot;
    return tot;
  };
  roots.forEach(rollup);
  return roots;
}

function buildFallbackTreeFromMeshes(meshes: HierarchyMeshInfo[]): HierarchyNode[] {
  const map = new Map<string, HierarchyNode>();
  const roots: HierarchyNode[] = [];
  for (const m of meshes) {
    const segs = (m.path || `/${m.name}`).split("/").filter(Boolean);
    let curList = roots;
    let prefix = "";
    let last: HierarchyNode | null = null;
    for (const s of segs) {
      prefix = `${prefix}/${s}`;
      let nd = map.get(prefix);
      if (!nd) {
        nd = {
          name: s,
          path: prefix,
          kind: "group",
          camera_count: 0,
          mesh_count: 0,
          face_count: 0,
          children: [],
        };
        map.set(prefix, nd);
        curList.push(nd);
      }
      curList = nd.children;
      last = nd;
    }
    if (last) {
      last.kind = "mesh";
      last.mesh = m;
    }
  }
  const rollup = (n: HierarchyNode): [number, number] => {
    let mc = n.mesh ? 1 : 0;
    let fc = n.mesh ? n.mesh.face_count : 0;
    for (const ch of n.children) {
      const [cm, cf] = rollup(ch);
      mc += cm;
      fc += cf;
    }
    n.mesh_count = mc;
    n.face_count = fc;
    return [mc, fc];
  };
  roots.forEach(rollup);
  return roots;
}

function collectDescendantMeshPaths(node: HierarchyNode): string[] {
  const out: string[] = [];
  if (node.mesh) out.push(node.path);
  for (const ch of node.children) {
    out.push(...collectDescendantMeshPaths(ch));
  }
  return out;
}

function nodeMatchesSearch(node: HierarchyNode, q: string): boolean {
  if (!q) return true;
  if (node.name.toLowerCase().includes(q) || node.path.toLowerCase().includes(q)) return true;
  return node.children.some((c) => nodeMatchesSearch(c, q));
}

export const CameraHierarchyModal: React.FC<{
  format: string;
  unitLabel: string;
  unitToCm: number;
  cameras: HierarchyCameraInfo[];
  hierarchy?: HierarchyNode[];
  currentCameraPath: string;
  onConfirm: (cameraPath: string) => Promise<void> | void;
  onClose: () => void;
}> = ({ format, unitLabel, unitToCm, cameras, hierarchy, currentCameraPath, onConfirm, onClose }) => {
  const [selectedPath, setSelectedPath] = useState<string>(
    currentCameraPath || cameras[0]?.path || ""
  );
  const [search, setSearch] = useState("");
  const [onlyCamBranches, setOnlyCamBranches] = useState(true);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [submitting, setSubmitting] = useState(false);

  const tree = useMemo(() => {
    if (hierarchy && hierarchy.length > 0) return hierarchy;
    return buildFallbackTreeFromCameras(cameras);
  }, [hierarchy, cameras]);

  const q = search.trim().toLowerCase();

  const toggleCollapse = (path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const handleApply = async (pathToUse = selectedPath) => {
    if (!pathToUse || submitting) return;
    setSubmitting(true);
    try {
      await onConfirm(pathToUse);
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  const renderNode = (node: HierarchyNode, depth: number): React.ReactNode => {
    if (onlyCamBranches && node.camera_count === 0) return null;
    if (!nodeMatchesSearch(node, q)) return null;

    const hasChildren = node.children.some(
      (ch) => (!onlyCamBranches || ch.camera_count > 0) && nodeMatchesSearch(ch, q)
    );
    const isCollapsed = collapsed.has(node.path) && !q;
    const isCam = Boolean(node.camera);
    const isChosen = isCam && selectedPath === node.path;

    const cam = node.camera;
    const fCnt = cam
      ? cam.frame_count ?? (cam.frames?.length === 2 ? cam.frames[1] - cam.frames[0] + 1 : 1)
      : 0;
    const travelMeters =
      cam && cam.travel_raw !== undefined ? (cam.travel_raw * unitToCm) / 100.0 : null;

    return (
      <div key={node.path}>
        <div
          className={`outliner-row ${isCam ? "outliner-selectable" : ""} ${
            isChosen ? "outliner-selected" : ""
          }`}
          style={{ paddingLeft: 10 + depth * 18 }}
          onClick={() => {
            if (isCam) setSelectedPath(node.path);
            else if (hasChildren) toggleCollapse(node.path);
          }}
          onDoubleClick={() => {
            if (isCam) handleApply(node.path);
          }}
        >
          <span
            className={`outliner-twisty ${isCollapsed ? "is-collapsed" : ""}`}
            onClick={(e) => {
              if (hasChildren) {
                e.stopPropagation();
                toggleCollapse(node.path);
              }
            }}
          >
            {hasChildren ? <IconChevronDown size={12} /> : null}
          </span>

          {isCam ? (
            <input
              type="radio"
              name="cam-hierarchy-pick"
              aria-label={node.path}
              checked={isChosen}
              onChange={() => setSelectedPath(node.path)}
              onClick={(e) => e.stopPropagation()}
            />
          ) : null}

          <span className={`outliner-icon ${isCam ? "is-leaf" : ""}`}>
            {isCam ? <IconCamera size={14} /> : <IconFolder size={14} />}
          </span>

          <span className="outliner-name" title={node.path}>
            {node.name}
          </span>

          {isCam && cam ? (
            <span className="outliner-meta">
              <span className="tag tag-info">{cam.focal_mm} mm</span>
              <span>
                {cam.frames?.length === 2
                  ? cam.frames[0] === cam.frames[1]
                    ? `单帧 (${cam.frames[0]})`
                    : `帧 ${cam.frames[0]}–${cam.frames[1]} (${fCnt}帧)`
                  : `${fCnt}帧`}
              </span>
              {travelMeters !== null && (
                <span className={travelMeters > 0.01 ? "is-moving" : "dim"}>
                  {travelMeters > 0.005 ? `位移 ${travelMeters.toFixed(2)} m` : "静止"}
                </span>
              )}
            </span>
          ) : (
            <span className="outliner-meta">
              {node.camera_count > 0 ? `含 ${node.camera_count} 台摄影机` : "组节点"}
            </span>
          )}
        </div>

        {hasChildren && !isCollapsed && (
          <div className="outliner-children">
            {node.children.map((ch) => renderNode(ch, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <Modal
      title={`选择层级摄影机（${format.toUpperCase()} · 共 ${cameras.length} 台）`}
      titleExtra={<span className="tag tag-info">单位: {unitLabel}</span>}
      width={680}
      maxHeight="82vh"
      onClose={onClose}
      footer={
        <>
          <div className="modal-footer-note" title={selectedPath}>
            当前选中：<strong style={{ color: "var(--accent-text)" }}>{selectedPath || "未选择"}</strong>
          </div>
          <div className="modal-footer-actions">
            <button type="button" className="btn" onClick={onClose}>
              取消
            </button>
            <button
              type="button"
              className={`btn btn-primary ${submitting ? "is-busy" : ""}`}
              disabled={!selectedPath || submitting}
              onClick={() => handleApply(selectedPath)}
            >
              {submitting && <span className="spinner" />}
              {submitting ? "正在切换..." : "确认选择此摄影机"}
            </button>
          </div>
        </>
      }
    >
      <div className="row row-wrap gap-12">
        <input
          type="text"
          className="grow"
          placeholder="搜索层级路径或摄影机名称..."
          aria-label="搜索层级路径或摄影机名称"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ minWidth: 200 }}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={onlyCamBranches}
            onChange={(e) => setOnlyCamBranches(e.target.checked)}
          />
          仅显示摄影机相关层级
        </label>
      </div>

      <div className="outliner-tree-box">{tree.map((r) => renderNode(r, 0))}</div>
    </Modal>
  );
};

export const SceneScanHierarchyModal: React.FC<{
  format: string;
  meshes: HierarchyMeshInfo[];
  hierarchy?: HierarchyNode[];
  selectedPaths?: string[];
  onConfirm: (selectedPaths: string[]) => Promise<void> | void;
  onClose: () => void;
}> = ({ format, meshes, hierarchy, selectedPaths, onConfirm, onClose }) => {
  const allMeshPaths = useMemo(() => meshes.map((m) => m.path), [meshes]);

  const [checkedSet, setCheckedSet] = useState<Set<string>>(() => {
    if (!selectedPaths || selectedPaths.length === 0) {
      return new Set(allMeshPaths);
    }
    const matched = allMeshPaths.filter((mp) =>
      selectedPaths.some((sp) => sp === "/" || mp === sp || mp.startsWith(sp.replace(/\/$/, "") + "/"))
    );
    return new Set(matched.length > 0 ? matched : allMeshPaths);
  });

  const [search, setSearch] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [submitting, setSubmitting] = useState(false);

  const tree = useMemo(() => {
    if (hierarchy && hierarchy.length > 0) return hierarchy;
    return buildFallbackTreeFromMeshes(meshes);
  }, [hierarchy, meshes]);

  const q = search.trim().toLowerCase();

  const selectedFaceCount = useMemo(() => {
    let tot = 0;
    for (const m of meshes) {
      if (checkedSet.has(m.path)) tot += m.face_count;
    }
    return tot;
  }, [meshes, checkedSet]);

  const toggleCollapse = (path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const toggleNodeMeshes = (node: HierarchyNode) => {
    const desc = collectDescendantMeshPaths(node);
    if (desc.length === 0) return;
    setCheckedSet((prev) => {
      const next = new Set(prev);
      const allChecked = desc.every((p) => next.has(p));
      if (allChecked) {
        desc.forEach((p) => next.delete(p));
      } else {
        desc.forEach((p) => next.add(p));
      }
      return next;
    });
  };

  const soloNodeMeshes = (node: HierarchyNode) => {
    const desc = collectDescendantMeshPaths(node);
    if (desc.length === 0) return;
    setCheckedSet(new Set(desc));
  };

  const handleApply = async () => {
    if (checkedSet.size === 0 || submitting) return;
    setSubmitting(true);
    try {
      // Empty list `[]` represents "all meshes selected" cleanly
      const payload =
        checkedSet.size >= allMeshPaths.length
          ? []
          : allMeshPaths.filter((p) => checkedSet.has(p));
      await onConfirm(payload);
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  const renderNode = (node: HierarchyNode, depth: number): React.ReactNode => {
    if (node.mesh_count === 0) return null;
    if (!nodeMatchesSearch(node, q)) return null;

    const descPaths = collectDescendantMeshPaths(node);
    const checkedCnt = descPaths.filter((p) => checkedSet.has(p)).length;
    const isAllChecked = descPaths.length > 0 && checkedCnt === descPaths.length;
    const isPartial = checkedCnt > 0 && checkedCnt < descPaths.length;

    const hasChildren = node.children.some((ch) => ch.mesh_count > 0 && nodeMatchesSearch(ch, q));
    const isCollapsed = collapsed.has(node.path) && !q;
    const isMesh = Boolean(node.mesh);

    return (
      <div key={node.path}>
        <div
          className={`outliner-row outliner-selectable ${isAllChecked ? "outliner-selected" : ""}`}
          style={{ paddingLeft: 10 + depth * 18 }}
          onClick={() => toggleNodeMeshes(node)}
        >
          <span
            className={`outliner-twisty ${isCollapsed ? "is-collapsed" : ""}`}
            onClick={(e) => {
              if (hasChildren) {
                e.stopPropagation();
                toggleCollapse(node.path);
              }
            }}
          >
            {hasChildren ? <IconChevronDown size={12} /> : null}
          </span>

          <input
            type="checkbox"
            aria-label={node.path}
            checked={isAllChecked}
            ref={(el) => {
              if (el) el.indeterminate = isPartial;
            }}
            onChange={() => toggleNodeMeshes(node)}
            onClick={(e) => e.stopPropagation()}
          />

          <span className={`outliner-icon ${isMesh ? "is-leaf" : ""}`}>
            {isMesh ? <IconCube size={14} /> : <IconFolder size={14} />}
          </span>

          <span className="outliner-name" title={node.path}>
            {node.name}
          </span>

          <span className="outliner-meta">
            {isMesh && node.mesh ? (
              <span>{node.mesh.face_count.toLocaleString()} 面</span>
            ) : (
              <span>
                {checkedCnt}/{descPaths.length} 物体 · {node.face_count.toLocaleString()} 面
              </span>
            )}
            <button
              type="button"
              className="btn btn-xs outliner-solo-btn"
              title="仅选择此层级节点"
              onClick={(e) => {
                e.stopPropagation();
                soloNodeMeshes(node);
              }}
            >
              仅选此项
            </button>
          </span>
        </div>

        {hasChildren && !isCollapsed && (
          <div className="outliner-children">
            {node.children.map((ch) => renderNode(ch, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <Modal
      title={`选择场景扫描层级物体（${format.toUpperCase()} · 共 ${meshes.length} 个网格）`}
      width={700}
      maxHeight="82vh"
      onClose={onClose}
      footer={
        <>
          <div className="modal-footer-note tnum">
            已勾选{" "}
            <strong style={{ color: checkedSet.size > 0 ? "var(--accent-text)" : "var(--err-text)" }}>
              {checkedSet.size} / {allMeshPaths.length}
            </strong>{" "}
            个网格物体（共 {selectedFaceCount.toLocaleString()} 面）
          </div>
          <div className="modal-footer-actions">
            <button type="button" className="btn" onClick={onClose}>
              取消
            </button>
            <button
              type="button"
              className={`btn btn-primary ${submitting ? "is-busy" : ""}`}
              disabled={checkedSet.size === 0 || submitting}
              onClick={handleApply}
            >
              {submitting && <span className="spinner" />}
              {submitting ? "正在更新扫描网格..." : "应用所选层级物体"}
            </button>
          </div>
        </>
      }
    >
      <div className="row row-wrap">
        <input
          type="text"
          className="grow"
          placeholder="搜索层级组或扫描物体名称..."
          aria-label="搜索层级组或扫描物体名称"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ minWidth: 200 }}
        />
        <button type="button" className="btn" onClick={() => setCheckedSet(new Set(allMeshPaths))}>
          全选 ({allMeshPaths.length})
        </button>
        <button type="button" className="btn" onClick={() => setCheckedSet(new Set())}>
          清除
        </button>
      </div>

      <div className="outliner-tree-box">{tree.map((r) => renderNode(r, 0))}</div>
    </Modal>
  );
};
