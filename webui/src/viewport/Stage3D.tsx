import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  DecodedCamera,
  DecodedCharacter,
  DecodedObject,
  DecodedScene,
  DecodedSceneScan,
  DetectedPerson,
  GradeParams,
  ManualEvaluatedFrameStatus,
  ObjectPromptItem,
  PlateMeta,
  SelectionState,
  TrackedObjectInfo,
  WorkMode,
  objectColor,
  personColor,
} from "../types";

export type ViewMode = "camera" | "persp";
export type CharacterDisplay = "both" | "mesh" | "skeleton" | "off";

interface StageProps {
  shotId: string;
  plateRev: string;
  frame: number;
  plate: PlateMeta | null;
  grade: GradeParams;
  bypassGrade: boolean;
  scene: DecodedScene | null;
  viewMode: ViewMode;
  displayMode: CharacterDisplay;
  overlayOpacity: number;
  autoHide: boolean;
  wipeEnabled?: boolean;
  wipeRatio?: number;
  onWipeRatioChange?: (ratio: number) => void;
  workMode?: WorkMode;
  showBoxes: boolean;
  showObjectMasks?: boolean;
  showSceneScan?: boolean;
  detectedPeople: DetectedPerson[];
  selectedPersonIds: Set<number>;
  selection: SelectionState;
  manualDrawMode?: boolean;
  activeManualPersonId?: number | null;
  manualStatusesByPerson?: Record<number, Record<number, ManualEvaluatedFrameStatus>>;
  onDrawManualBox?: (frame: number, box: [number, number, number, number]) => void;
  objectPrompts?: ObjectPromptItem[];
  activeObjectId?: number;
  objectClickLabel?: 0 | 1;
  trackedObjects?: TrackedObjectInfo[];
  onPickPoint?: (frame: number, xPx: number, yPx: number) => void;
  onPickObjectPoint?: (frame: number, xPx: number, yPx: number, label: 0 | 1) => void;
  onToggleBoxPerson?: (personId: number) => void;
  /** Solved people chosen for export; the others are drawn dimmed. Undefined = everybody. */
  exportPersonIds?: Set<number>;
  /** A click on a solved person in the view (camera view or perspective) toggles that person's export choice. */
  onToggleExportPerson?: (personId: number) => void;
  gateZoom: number;
  gatePan: [number, number];
  onGateChange: (zoom: number, pan: [number, number]) => void;
}

// ------------------------------------------------------------------ ImageBitmap Backplate Texture Cache

const bitmapCache = new Map<string, THREE.Texture>();
const inflight = new Map<string, Promise<THREE.Texture | null>>();

function loadFrameTexture(shotId: string, plateRev: string, frame: number): Promise<THREE.Texture | null> {
  const key = `${shotId}:${plateRev}:${frame}`;
  const hit = bitmapCache.get(key);
  if (hit) return Promise.resolve(hit);
  const pend = inflight.get(key);
  if (pend) return pend;

  const p = fetch(`/api/shots/${encodeURIComponent(shotId)}/frames/${frame}?v=${encodeURIComponent(plateRev)}`)
    .then((r) => (r.ok ? r.blob() : null))
    .then((blob) => (blob ? createImageBitmap(blob) : null))
    .then((bmp) => {
      inflight.delete(key);
      if (!bmp) return null;
      const tex = new THREE.Texture(bmp as unknown as HTMLImageElement);
      tex.flipY = false;
      tex.repeat.set(1, -1);
      tex.offset.set(0, 1);
      tex.colorSpace = THREE.NoColorSpace;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = false;
      tex.needsUpdate = true;
      if (bitmapCache.size > 180) {
        const oldest = bitmapCache.keys().next().value;
        if (oldest) {
          bitmapCache.get(oldest)?.dispose();
          bitmapCache.delete(oldest);
        }
      }
      bitmapCache.set(key, tex);
      return tex;
    })
    .catch(() => {
      inflight.delete(key);
      return null;
    });

  inflight.set(key, p);
  return p;
}

function buildDefaultOriginCam(plate: PlateMeta): DecodedCamera {
  const w = plate.width || 1920;
  const h = plate.height || 1080;
  const hap = 36.0;
  const vap = (hap * h) / Math.max(w, 1);
  const fMm = (hap * Math.hypot(w, h)) / Math.max(w, 1);
  const world = new Float32Array(16);
  world[0] = 1;
  world[5] = 1;
  world[10] = 1;
  world[15] = 1;
  const m = new Map<number, number>();
  m.set(1, 0);
  return {
    role: "origin",
    name: "origin_cam",
    path: "/origin_cam",
    frames: [1],
    frameToIndex: m,
    width: w,
    height: h,
    pixel_aspect: 1.0,
    overscan: [0, 0, 0, 0],
    fps: plate.fps || 24,
    world,
    focal_mm: new Float32Array([fMm]),
    h_aperture_mm: new Float32Array([hap]),
    v_aperture_mm: new Float32Array([vap]),
    center_mm: new Float32Array([0, 0]),
  };
}

function sampleCameraAt(cam: DecodedCamera, frame: number) {
  let idx = cam.frameToIndex.get(frame);
  if (idx === undefined) {
    if (frame <= cam.frames[0]) idx = 0;
    else if (frame >= cam.frames[cam.frames.length - 1]) idx = cam.frames.length - 1;
    else {
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < cam.frames.length; i++) {
        const d = Math.abs(cam.frames[i] - frame);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      idx = best;
    }
  }
  const m = new THREE.Matrix4().fromArray(cam.world, idx * 16);
  const focal = cam.focal_mm[Math.min(idx, cam.focal_mm.length - 1)];
  const hap = cam.h_aperture_mm[Math.min(idx, cam.h_aperture_mm.length - 1)];
  const vap = cam.v_aperture_mm[Math.min(idx, cam.v_aperture_mm.length - 1)];
  const cIdx = Math.max(0, Math.min(idx, cam.center_mm.length / 2 - 1));
  const cxMm = cam.center_mm[cIdx * 2] || 0;
  const cyMm = cam.center_mm[cIdx * 2 + 1] || 0;
  const shiftX = hap > 0 ? cxMm / hap : 0;
  const shiftY = vap > 0 ? cyMm / vap : 0;
  return { matrix: m, focal, hap, vap, shiftX, shiftY };
}

function computeGateRect(viewW: number, viewH: number, aspect: number) {
  if (viewW <= 0 || viewH <= 0) return { x: 0, y: 0, w: 1, h: 1 };
  const viewAspect = viewW / viewH;
  if (viewAspect > aspect) {
    const h = viewH * 0.94;
    const w = h * aspect;
    return { x: (viewW - w) * 0.5, y: (viewH - h) * 0.5, w, h };
  } else {
    const w = viewW * 0.94;
    const h = w / aspect;
    return { x: (viewW - w) * 0.5, y: (viewH - h) * 0.5, w, h };
  }
}

// ------------------------------------------------------------------ Look-Through Camera + Exact WebGL Grade Backplate

const LookThroughController: React.FC<{
  cam: DecodedCamera;
  frame: number;
  plateW: number;
  plateH: number;
  shotId: string;
  plateRev: string;
  hasPlate: boolean;
  isLinearOcio: boolean;
  grade: GradeParams;
  bypassGrade: boolean;
  gateZoom: number;
  gatePan: [number, number];
}> = ({ cam, frame, plateW, plateH, shotId, plateRev, hasPlate, isLinearOcio, grade, bypassGrade, gateZoom, gatePan }) => {
  const { camera, size } = useThree();
  const planeRef = useRef<THREE.Mesh>(null);
  const [tex, setTex] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    if (!hasPlate || !shotId) {
      setTex(null);
      return;
    }
    let alive = true;
    loadFrameTexture(shotId, plateRev, frame).then((t) => {
      if (alive && t) setTex(t);
    });
    loadFrameTexture(shotId, plateRev, frame + 1);
    loadFrameTexture(shotId, plateRev, frame + 2);
    return () => {
      alive = false;
    };
  }, [shotId, plateRev, frame, hasPlate]);

  const shaderMat = useMemo(() => {
    return new THREE.ShaderMaterial({
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      uniforms: {
        uMap: { value: null },
        uIsLinearOcio: { value: 0.0 },
        uExposure: { value: 0.0 },
        uBrightness: { value: 0.0 },
        uContrast: { value: 1.0 },
        uSaturation: { value: 1.0 },
        uGamma: { value: 1.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform float uIsLinearOcio;
        uniform float uExposure;
        uniform float uBrightness;
        uniform float uContrast;
        uniform float uSaturation;
        uniform float uGamma;
        varying vec2 vUv;

        vec3 srgbToLinear(vec3 s) {
          bvec3 lo = lessThanEqual(s, vec3(0.04045));
          vec3 a = s / 12.92;
          vec3 b = pow((max(s, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
          return mix(b, a, vec3(lo));
        }

        vec3 linearToSrgb(vec3 l) {
          vec3 cl = clamp(l, 0.0, 1.0);
          bvec3 lo = lessThanEqual(cl, vec3(0.0031308));
          vec3 a = cl * 12.92;
          vec3 b = 1.055 * pow(max(cl, vec3(1e-8)), vec3(1.0 / 2.4)) - 0.055;
          return mix(b, a, vec3(lo));
        }

        vec3 applyLinearOcioExposure(vec3 srgbIn, float ev) {
          vec3 y = clamp(srgbToLinear(srgbIn), 0.0, 0.985);
          vec3 qa = vec3(2.51) - 2.43 * y;
          vec3 qb = vec3(0.03) - 0.59 * y;
          vec3 qc = -0.14 * y;
          vec3 x = (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, vec3(0.0)))) / (2.0 * qa);
          vec3 xEv = x * exp2(ev);
          vec3 mapped = clamp((xEv * (2.51 * xEv + 0.03)) / (xEv * (2.43 * xEv + 0.59) + 0.14), 0.0, 1.0);
          return linearToSrgb(mapped);
        }

        void main() {
          vec2 uvFlip = vec2(vUv.x, 1.0 - vUv.y);
          vec4 texel = texture2D(uMap, uvFlip);
          vec3 baseRgb = texel.rgb;
          vec3 c;
          if (uIsLinearOcio > 0.5 && abs(uExposure) > 1e-5) {
            c = applyLinearOcioExposure(baseRgb, uExposure) + vec3(uBrightness);
          } else {
            c = baseRgb * exp2(uExposure) + vec3(uBrightness);
          }
          c = (c - vec3(0.5)) * uContrast + vec3(0.5);
          c = pow(max(c, vec3(0.0)), vec3(1.0 / max(uGamma, 0.05)));
          float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
          c = mix(vec3(luma), c, uSaturation);
          gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
        }
      `,
    });
  }, []);

  useFrame(() => {
    const persp = camera as THREE.PerspectiveCamera;
    const s = sampleCameraAt(cam, frame);

    persp.matrixAutoUpdate = false;
    persp.matrix.copy(s.matrix);
    persp.matrixWorld.copy(s.matrix);
    persp.matrixWorldInverse.copy(s.matrix).invert();

    const effW = plateW > 0 ? plateW : cam.width;
    const effH = plateH > 0 ? plateH : cam.height;
    const aspect =
      effH > 0 ? (effW * cam.pixel_aspect) / effH : s.hap / Math.max(s.vap, 1e-4);
    const vHeightMm = s.hap / Math.max(aspect, 1e-4);
    const fovDeg = THREE.MathUtils.radToDeg(2.0 * Math.atan(vHeightMm / (2.0 * Math.max(s.focal, 0.1))));

    persp.fov = fovDeg;
    persp.near = 5.0;
    persp.far = 250000.0;
    persp.aspect = aspect;

    const gate = computeGateRect(size.width, size.height, aspect);
    const gw = gate.w * gateZoom;
    const gh = gate.h * gateZoom;
    const gx = size.width * 0.5 - gw * 0.5 + gatePan[0];
    const gy = size.height * 0.5 - gh * 0.5 + gatePan[1];
    const subX = -gx - s.shiftX * gw;
    const subY = -gy + s.shiftY * gh;
    persp.setViewOffset(gw, gh, subX, subY, size.width, size.height);
    persp.updateProjectionMatrix();

    if (planeRef.current && tex) {
      const depth = 60000.0;
      const planeH = 2.0 * depth * Math.tan(THREE.MathUtils.degToRad(fovDeg * 0.5));
      const planeW = planeH * aspect;
      const localPos = new THREE.Vector3(-s.shiftX * planeW, -s.shiftY * planeH, -depth);
      planeRef.current.matrixAutoUpdate = false;
      const localM = new THREE.Matrix4().compose(
        localPos,
        new THREE.Quaternion(),
        new THREE.Vector3(planeW, planeH, 1.0)
      );
      planeRef.current.matrix.copy(s.matrix).multiply(localM);
      planeRef.current.matrixWorld.copy(planeRef.current.matrix);

      shaderMat.uniforms.uMap.value = tex;
      shaderMat.uniforms.uIsLinearOcio.value = isLinearOcio ? 1.0 : 0.0;
      shaderMat.uniforms.uExposure.value = bypassGrade ? 0.0 : grade.exposure;
      shaderMat.uniforms.uBrightness.value = bypassGrade ? 0.0 : grade.brightness;
      shaderMat.uniforms.uContrast.value = bypassGrade ? 1.0 : (grade.contrast ?? 1.0);
      shaderMat.uniforms.uSaturation.value = bypassGrade ? 1.0 : (grade.saturation ?? 1.0);
      shaderMat.uniforms.uGamma.value = bypassGrade ? 1.0 : grade.gamma;
    }
  });

  useEffect(() => {
    return () => {
      const persp = camera as THREE.PerspectiveCamera;
      persp.clearViewOffset();
      persp.matrixAutoUpdate = true;
      persp.updateProjectionMatrix();
    };
  }, [camera]);

  if (!hasPlate || !tex) return null;
  return (
    <mesh ref={planeRef} renderOrder={-100} frustumCulled={false} material={shaderMat}>
      <planeGeometry args={[1, 1]} />
    </mesh>
  );
};

// ------------------------------------------------------------------ Maya-Style 3-Button Perspective Controller

const MayaPerspectiveController: React.FC<{
  scene: DecodedScene | null;
  shotId: string;
}> = ({ scene, shotId }) => {
  const { camera, size, gl } = useThree();
  const controlsRef = useRef<any>(null);
  const framedSigRef = useRef<string>("");

  useEffect(() => {
    const persp = camera as THREE.PerspectiveCamera;
    persp.clearViewOffset();
    persp.matrixAutoUpdate = true;
    persp.fov = 42;
    persp.near = 2.0;
    persp.far = 250000.0;
    persp.aspect = size.width / Math.max(size.height, 1);
    persp.up.set(0, 1, 0);
    persp.updateProjectionMatrix();
  }, [camera, size.width, size.height]);

  const frameScene = useMemo(() => {
    return () => {
      const persp = camera as THREE.PerspectiveCamera;
      const box = new THREE.Box3();
      if (scene) {
        for (const ch of scene.characters) {
          if (ch.anim_world.length >= 16) {
            const nJ = ch.parents.length;
            const visIndices = ch.frames
              .map((_, idx) => idx)
              .filter((idx) => !ch.visible || ch.visible[idx] > 0);
            const pool = visIndices.length > 0 ? visIndices : [0];
            const sampleFrames = [pool[0], pool[Math.floor(pool.length / 2)], pool[pool.length - 1]];
            for (const fi of sampleFrames) {
              if (fi >= 0 && fi < ch.frames.length) {
                const o = fi * nJ * 16;
                const p = new THREE.Vector3(ch.anim_world[o + 12], ch.anim_world[o + 13], ch.anim_world[o + 14]);
                box.expandByPoint(p.clone().add(new THREE.Vector3(90, 100, 90)));
                box.expandByPoint(p.clone().add(new THREE.Vector3(-90, -100, -90)));
              }
            }
          }
        }
        for (const obj of scene.objects || []) {
          if (obj.world.length >= 16) {
            const visIndices = obj.frames
              .map((_, idx) => idx)
              .filter((idx) => !obj.visible || obj.visible[idx] > 0);
            const pool = visIndices.length > 0 ? visIndices : [0];
            const sampleFrames = [pool[0], pool[Math.floor(pool.length / 2)], pool[pool.length - 1]];
            for (const fi of sampleFrames) {
              if (fi >= 0 && fi < obj.frames.length) {
                const o = fi * 16;
                const p = new THREE.Vector3(obj.world[o + 12], obj.world[o + 13], obj.world[o + 14]);
                box.expandByPoint(p.clone().add(new THREE.Vector3(60, 60, 60)));
                box.expandByPoint(p.clone().add(new THREE.Vector3(-60, -60, -60)));
              }
            }
          }
        }
        for (const c of scene.cameras) {
          for (let i = 0; i < c.frames.length; i++) {
            const o = i * 16;
            box.expandByPoint(new THREE.Vector3(c.world[o + 12], c.world[o + 13], c.world[o + 14]));
          }
        }
        if (scene.scene_scan && scene.scene_scan.vertices.length >= 9 && box.isEmpty()) {
          const sv = scene.scene_scan.vertices;
          const step = Math.max(3, Math.floor(sv.length / 600) * 3);
          for (let i = 0; i < sv.length; i += step) {
            box.expandByPoint(new THREE.Vector3(sv[i], sv[i + 1], sv[i + 2]));
          }
        }
      }
      const center = new THREE.Vector3(0, 90, -250);
      let radius = 260;
      if (!box.isEmpty()) {
        const sphere = new THREE.Sphere();
        box.getBoundingSphere(sphere);
        center.copy(sphere.center);
        radius = Math.max(sphere.radius, 120);
      }
      const dist = radius * 2.15;
      const offset = new THREE.Vector3(0.68, 0.42, 0.72).normalize().multiplyScalar(dist);
      persp.position.copy(center).add(offset);
      persp.up.set(0, 1, 0);
      persp.lookAt(center);
      if (controlsRef.current) {
        controlsRef.current.target.copy(center);
        controlsRef.current.update();
      }
    };
  }, [scene, camera]);

  useEffect(() => {
    if (!scene) return;
    const camRole = scene.cameras[0]?.role || "none";
    const camName = scene.cameras[0]?.name || "";
    const sig = `${shotId}:${camRole}:${camName}:${scene.characters.length}:${scene.objects?.length || 0}`;
    if (framedSigRef.current === sig) return;
    framedSigRef.current = sig;
    frameScene();
  }, [scene, shotId, frameScene]);

  // Maya Right-Mouse-Button Dolly (drag right or up = dolly in; drag left or down = dolly out) + 'F' key frame!
  useEffect(() => {
    const el = gl.domElement;
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 2 || !controlsRef.current) return;
      e.preventDefault();
      let lastX = e.clientX;
      let lastY = e.clientY;
      const onMove = (ev: PointerEvent) => {
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        const delta = dx - dy;
        if (!delta) return;
        const ctrl = controlsRef.current;
        const persp = ctrl.object as THREE.PerspectiveCamera;
        const factor = Math.exp(-delta * 0.0052);
        const off = persp.position.clone().sub(ctrl.target);
        const nextDist = THREE.MathUtils.clamp(off.length() * factor, 15.0, 200000.0);
        persp.position.copy(ctrl.target).addScaledVector(off.normalize(), nextDist);
        ctrl.update();
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Alt") {
        // Prevent Windows Chrome/Edge from stealing focus to the browser menu bar during Maya Alt+Mouse navigation
        e.preventDefault();
      }
      if (
        (e.key === "f" || e.key === "F") &&
        !(e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement)
      ) {
        frameScene();
      }
    };

    el.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [gl, frameScene]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.1}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
        RIGHT: -1 as unknown as THREE.MOUSE,
      }}
    />
  );
};

// ------------------------------------------------------------------ Camera Frustum + Motion Trajectory Path Line

const CameraGizmo: React.FC<{ cam: DecodedCamera; frame: number }> = ({ cam, frame }) => {
  const pathPoints = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < cam.frames.length; i++) {
      const o = i * 16;
      pts.push([cam.world[o + 12], cam.world[o + 13], cam.world[o + 14]]);
    }
    return pts;
  }, [cam]);

  const s = sampleCameraAt(cam, frame);
  const aspect = cam.height > 0 ? (cam.width * cam.pixel_aspect) / cam.height : s.hap / Math.max(s.vap, 1e-4);
  const depth = 48.0;
  const h = (depth * (s.hap / Math.max(aspect, 1e-4))) / Math.max(s.focal, 1.0);
  const w = h * aspect;
  const cx = -s.shiftX * w;
  const cy = -s.shiftY * h;

  const frustumLines = useMemo(() => {
    const c0: [number, number, number] = [0, 0, 0];
    const tl: [number, number, number] = [cx - w * 0.5, cy + h * 0.5, -depth];
    const tr: [number, number, number] = [cx + w * 0.5, cy + h * 0.5, -depth];
    const br: [number, number, number] = [cx + w * 0.5, cy - h * 0.5, -depth];
    const bl: [number, number, number] = [cx - w * 0.5, cy - h * 0.5, -depth];
    const upTip: [number, number, number] = [cx, cy + h * 0.68, -depth];
    return [c0, tl, c0, tr, c0, br, c0, bl, tl, tr, tr, br, br, bl, bl, tl, tl, upTip, upTip, tr];
  }, [w, h, cx, cy]);

  const pos = new THREE.Vector3().setFromMatrixPosition(s.matrix);
  const quat = new THREE.Quaternion().setFromRotationMatrix(s.matrix);
  const isOrigin = cam.role === "origin";

  return (
    <group>
      {!isOrigin && pathPoints.length > 1 && (
        <Line points={pathPoints} color="#38bdf8" lineWidth={2.4} />
      )}
      <group position={pos} quaternion={quat}>
        <Line points={frustumLines} color={isOrigin ? "#34d399" : "#facc15"} lineWidth={1.9} segments />
      </group>
    </group>
  );
};

// ------------------------------------------------------------------ Screen-Space Vertical Wipe Clipping Plane

const WipePlaneController: React.FC<{
  wipePlane: THREE.Plane;
  wipeEnabled: boolean;
  wipeRatio: number;
}> = ({ wipePlane, wipeEnabled, wipeRatio }) => {
  const projScreen = useMemo(() => new THREE.Matrix4(), []);

  useFrame(({ camera }) => {
    if (!wipeEnabled) {
      wipePlane.normal.set(0, 1, 0);
      wipePlane.constant = 1e12;
      return;
    }
    camera.updateMatrixWorld();
    projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = projScreen.elements;
    // Keep fragments strictly on the LEFT of wipeRatio (`x_clip / w_clip <= xNdc`):
    // drag all the way LEFT (`wipeRatio = 0`) -> all 3D models disappear (only background remains);
    // drag RIGHT (`0 -> 1`) -> 3D models progressively reveal across the tracked objects!
    const xNdc = Math.max(-1.0, Math.min(1.0, wipeRatio * 2.0 - 1.0));
    const nx = xNdc * e[3] - e[0];
    const ny = xNdc * e[7] - e[4];
    const nz = xNdc * e[11] - e[8];
    const c = xNdc * e[15] - e[12];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-9) {
      wipePlane.normal.set(nx / len, ny / len, nz / len);
      wipePlane.constant = c / len;
    }
  });

  return null;
};

// ------------------------------------------------------------------ Clicking a solved person in the view

interface PickableCharacter {
  mesh: THREE.SkinnedMesh;
  ch: DecodedCharacter;
  frameIndex: number;
}

interface PickBridge {
  camera: THREE.Camera;
  dom: HTMLCanvasElement;
}

/** Hands the live camera and canvas of the 3D view to the code outside the canvas that handles mouse clicks. */
const PickBridgeProbe: React.FC<{ bridgeRef: React.MutableRefObject<PickBridge | null> }> = ({ bridgeRef }) => {
  const { camera, gl } = useThree();
  useEffect(() => {
    bridgeRef.current = { camera, dom: gl.domElement };
    return () => {
      bridgeRef.current = null;
    };
  }, [camera, gl, bridgeRef]);
  return null;
};

const _pickRay = new THREE.Raycaster();
const _pickSphere = new THREE.Sphere();

/** The solved person under a click (nearest surface along the view ray), or null. */
function pickCharacterAt(
  bridge: PickBridge | null,
  registry: Map<number, PickableCharacter>,
  clientX: number,
  clientY: number
): number | null {
  if (!bridge || registry.size === 0) return null;
  const rect = bridge.dom.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1
  );
  bridge.camera.updateMatrixWorld();
  _pickRay.setFromCamera(ndc, bridge.camera);
  let bestId: number | null = null;
  let bestDist = Infinity;
  registry.forEach((item, pid) => {
    const { ch, frameIndex, mesh } = item;
    const nJ = ch.parents.length;
    const base = frameIndex * nJ * 16;
    // sphere around this frame's joints (+ body thickness)
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let j = 0; j < nJ; j++) {
      cx += ch.anim_world[base + j * 16 + 12];
      cy += ch.anim_world[base + j * 16 + 13];
      cz += ch.anim_world[base + j * 16 + 14];
    }
    cx /= nJ;
    cy /= nJ;
    cz /= nJ;
    let r2 = 0;
    for (let j = 0; j < nJ; j++) {
      const ex = ch.anim_world[base + j * 16 + 12] - cx;
      const ey = ch.anim_world[base + j * 16 + 13] - cy;
      const ez = ch.anim_world[base + j * 16 + 14] - cz;
      r2 = Math.max(r2, ex * ex + ey * ey + ez * ez);
    }
    _pickSphere.center.set(cx, cy, cz);
    _pickSphere.radius = Math.sqrt(r2) + 25;
    if (!_pickRay.ray.intersectsSphere(_pickSphere)) return;
    // exact test against the posed surface; the sphere above also serves as the mesh's own early-out
    mesh.boundingSphere = _pickSphere.clone();
    (mesh as unknown as { boundingBox: THREE.Box3 | null }).boundingBox = null;
    const hits = _pickRay.intersectObject(mesh, false);
    if (hits.length > 0 && hits[0].distance < bestDist) {
      bestDist = hits[0].distance;
      bestId = pid;
    }
  });
  return bestId;
}

// ------------------------------------------------------------------ GPU SkinnedMesh + BoneFigure Character

const CharacterRigView: React.FC<{
  ch: DecodedCharacter;
  frame: number;
  displayMode: CharacterDisplay;
  opacity: number;
  autoHide: boolean;
  clipPlanes: THREE.Plane[];
  dimmed?: boolean;
  pickRegistry?: Map<number, PickableCharacter>;
}> = ({ ch, frame, displayMode, opacity, autoHide, clipPlanes, dimmed = false, pickRegistry }) => {
  const color = personColor(ch.person_id);
  const nJoints = ch.parents.length;

  const rigObj = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(ch.rest_points, 3));
    geom.setIndex(new THREE.BufferAttribute(ch.faces, 1));
    geom.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(ch.skin_indices, 4));
    geom.setAttribute("skinWeight", new THREE.Float32BufferAttribute(ch.skin_weights, 4));
    geom.computeVertexNormals();

    const bones: THREE.Bone[] = [];
    const boneInverses: THREE.Matrix4[] = [];
    for (let j = 0; j < nJoints; j++) {
      const b = new THREE.Bone();
      b.name = ch.joint_names[j] || `joint_${j}`;
      b.matrixAutoUpdate = false;
      bones.push(b);
      boneInverses.push(new THREE.Matrix4().fromArray(ch.inv_bind, j * 16));
    }
    const boneRoot = new THREE.Group();
    bones.forEach((b) => boneRoot.add(b));

    const skeleton = new THREE.Skeleton(bones, boneInverses);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      roughness: 0.45,
      metalness: 0.08,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      clippingPlanes: clipPlanes,
    });
    const skinned = new THREE.SkinnedMesh(geom, mat);
    skinned.frustumCulled = false;
    skinned.bind(skeleton, new THREE.Matrix4());

    const pairs: Array<[number, number]> = [];
    for (let j = 0; j < nJoints; j++) {
      if (ch.parents[j] >= 0) pairs.push([ch.parents[j], j]);
    }
    const lineGeo = new THREE.BufferGeometry();
    const linePos = new Float32Array(pairs.length * 6);
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: new THREE.Color("#ffffff"),
      depthTest: false,
      transparent: true,
      opacity: 0.92,
      clippingPlanes: clipPlanes,
    });
    const boneLines = new THREE.LineSegments(lineGeo, lineMat);
    boneLines.frustumCulled = false;
    boneLines.renderOrder = 20;

    return { boneRoot, bones, skinned, mat, pairs, lineGeo, boneLines };
  }, [ch, color, nJoints, clipPlanes]);

  const fIdx = ch.frameToIndex.get(frame);
  const visible = fIdx !== undefined && (!autoHide || !ch.visible || ch.visible[fIdx] > 0);

  // What a click in the view needs to find this person: the mesh (posed by the bones below) and a sphere around
  // the joints of the current frame, so people the click cannot hit are skipped without touching their vertices.
  useEffect(() => {
    if (!pickRegistry) return;
    if (!visible || fIdx === undefined) {
      pickRegistry.delete(ch.person_id);
      return;
    }
    pickRegistry.set(ch.person_id, { mesh: rigObj.skinned, ch, frameIndex: fIdx });
    return () => {
      pickRegistry.delete(ch.person_id);
    };
  }, [pickRegistry, visible, fIdx, rigObj, ch]);

  useFrame(() => {
    if (!visible || fIdx === undefined) return;
    const base = fIdx * nJoints * 16;
    for (let j = 0; j < nJoints; j++) {
      const b = rigObj.bones[j];
      b.matrix.fromArray(ch.anim_world, base + j * 16);
      b.matrixWorld.copy(b.matrix);
    }
    // people not chosen for export stay in the view, grey and faint, so they can still be clicked back in
    rigObj.mat.color.set(dimmed ? "#7b8494" : color);
    rigObj.mat.opacity = dimmed ? Math.min(opacity, 0.85) * 0.32 : opacity;
    rigObj.mat.depthWrite = !dimmed && opacity >= 0.95;
    (rigObj.boneLines.material as THREE.LineBasicMaterial).opacity = dimmed ? 0.22 : 0.92;

    if (displayMode !== "mesh" && rigObj.pairs.length > 0) {
      const arr = (rigObj.lineGeo.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
      for (let k = 0; k < rigObj.pairs.length; k++) {
        const [p, c] = rigObj.pairs[k];
        const po = base + p * 16;
        const co = base + c * 16;
        const out = k * 6;
        arr[out] = ch.anim_world[po + 12];
        arr[out + 1] = ch.anim_world[po + 13];
        arr[out + 2] = ch.anim_world[po + 14];
        arr[out + 3] = ch.anim_world[co + 12];
        arr[out + 4] = ch.anim_world[co + 13];
        arr[out + 5] = ch.anim_world[co + 14];
      }
      rigObj.lineGeo.getAttribute("position").needsUpdate = true;
    }
  });

  if (!visible) return null;

  return (
    <group>
      <primitive object={rigObj.boneRoot} />
      {displayMode !== "skeleton" && <primitive object={rigObj.skinned} />}
      {displayMode !== "mesh" && <primitive object={rigObj.boneLines} />}
    </group>
  );
};

// ------------------------------------------------------------------ Parent-Transform Rigid 3D Proxy Object

const RigidObjectMeshView: React.FC<{
  obj: DecodedObject;
  frame: number;
  displayMode: CharacterDisplay;
  opacity: number;
  autoHide: boolean;
  clipPlanes: THREE.Plane[];
}> = ({ obj, frame, displayMode, opacity, autoHide, clipPlanes }) => {
  const color = objectColor(obj.object_id);
  const groupRef = useRef<THREE.Group>(null);

  const meshPack = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(obj.rest_points, 3));
    geom.setIndex(new THREE.BufferAttribute(obj.faces, 1));
    geom.computeVertexNormals();

    // Crisp studio clay shading in the object's distinct color so the 3D surface is clearly visible over the plate
    // (avoiding dark photographic vertex colors that camouflage into the background video).
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      roughness: 0.36,
      metalness: 0.08,
      flatShading: false,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      clippingPlanes: clipPlanes,
    });

    const wireOverlayMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color("#ffffff"),
      wireframe: true,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      clippingPlanes: clipPlanes,
    });

    const wireOnlyMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color),
      wireframe: true,
      transparent: true,
      opacity: 0.62,
      depthWrite: false,
      clippingPlanes: clipPlanes,
    });

    return { geom, mat, wireOverlayMat, wireOnlyMat };
  }, [obj, color, clipPlanes]);

  const fIdx = obj.frameToIndex.get(frame);
  const visible = fIdx !== undefined && (!autoHide || !obj.visible || obj.visible[fIdx] > 0);

  const syncTransform = () => {
    if (!visible || fIdx === undefined || !groupRef.current) return;
    groupRef.current.matrixAutoUpdate = false;
    groupRef.current.matrix.fromArray(obj.world, fIdx * 16);
    groupRef.current.matrixWorld.copy(groupRef.current.matrix);
    meshPack.mat.opacity = opacity;
    meshPack.mat.depthWrite = opacity >= 0.92;
  };

  useEffect(() => {
    syncTransform();
  });

  useFrame(() => {
    syncTransform();
  });

  if (!visible) return null;

  return (
    <group ref={groupRef}>
      {displayMode !== "skeleton" && <mesh geometry={meshPack.geom} material={meshPack.mat} frustumCulled={false} />}
      {displayMode === "both" && (
        <mesh geometry={meshPack.geom} material={meshPack.wireOverlayMat} frustumCulled={false} />
      )}
      {displayMode === "skeleton" && (
        <mesh geometry={meshPack.geom} material={meshPack.wireOnlyMat} frustumCulled={false} />
      )}
    </group>
  );
};

// ------------------------------------------------------------------ Static World-Space Scene Scan Reference Mesh

const SceneScanMeshView: React.FC<{
  scan: DecodedSceneScan;
  isCameraMode: boolean;
  clipPlanes: THREE.Plane[];
}> = ({ scan, isCameraMode, clipPlanes }) => {
  const pack = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(scan.vertices, 3));
    geom.setIndex(new THREE.BufferAttribute(scan.faces, 1));
    geom.computeVertexNormals();

    const clayMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color("#475569"),
      roughness: 0.78,
      metalness: 0.05,
      transparent: true,
      opacity: 0.44,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 2,
      clippingPlanes: clipPlanes,
    });

    const wirePerspMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color("#38bdf8"),
      wireframe: true,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
      clippingPlanes: clipPlanes,
    });

    const wireCamMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color("#22d3ee"),
      wireframe: true,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
      clippingPlanes: clipPlanes,
    });

    return { geom, clayMat, wirePerspMat, wireCamMat };
  }, [scan, clipPlanes]);

  if (isCameraMode) {
    return <mesh geometry={pack.geom} material={pack.wireCamMat} frustumCulled={false} renderOrder={-10} />;
  }
  return (
    <group>
      <mesh geometry={pack.geom} material={pack.clayMat} frustumCulled={false} renderOrder={-12} />
      <mesh geometry={pack.geom} material={pack.wirePerspMat} frustumCulled={false} renderOrder={-11} />
    </group>
  );
};

// ------------------------------------------------------------------ Outer Stage Wrapper + Maya 2D Gate Controls + Boxes

export const Stage3D: React.FC<StageProps> = ({
  shotId,
  plateRev,
  frame,
  plate,
  grade,
  bypassGrade,
  scene,
  viewMode,
  displayMode,
  overlayOpacity,
  autoHide,
  wipeEnabled = false,
  wipeRatio = 0.5,
  onWipeRatioChange,
  workMode = "human",
  showBoxes,
  showObjectMasks = true,
  showSceneScan = true,
  detectedPeople,
  selectedPersonIds,
  selection,
  manualDrawMode = false,
  activeManualPersonId = null,
  manualStatusesByPerson = {},
  onDrawManualBox,
  objectPrompts = [],
  activeObjectId = 1,
  objectClickLabel = 1,
  trackedObjects = [],
  onPickPoint,
  onPickObjectPoint,
  onToggleBoxPerson,
  exportPersonIds,
  onToggleExportPerson,
  gateZoom,
  gatePan,
  onGateChange,
}) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [wrapSize, setWrapSize] = useState({ w: 960, h: 540 });
  const wipePlane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 1e12), []);
  const clipPlanes = useMemo(() => [wipePlane], [wipePlane]);
  const wipeDraggingRef = useRef<boolean>(false);
  const pickBridgeRef = useRef<PickBridge | null>(null);
  const pickRegistry = useMemo(() => new Map<number, PickableCharacter>(), []);
  const click3dRef = useRef<{ x: number; y: number } | null>(null);
  const [liveDrawBox, setLiveDrawBox] = useState<[number, number, number, number] | null>(null);
  const drag2dRef = useRef<{
    kind: "pan" | "zoom" | "click" | "wipe" | "draw_box";
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    pan0: [number, number];
    zoom0: number;
    plateStart?: [number, number];
  } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        setWrapSize({ w: e.contentRect.width, h: e.contentRect.height });
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const sceneCam = scene?.cameras[0] || null;
  const fallbackOriginCam = useMemo(() => (plate ? buildDefaultOriginCam(plate) : null), [plate]);
  const activeLookCam = viewMode === "camera" ? sceneCam || fallbackOriginCam : null;

  const plateW = plate?.width || sceneCam?.width || 1920;
  const plateH = plate?.height || sceneCam?.height || 1080;
  const aspect = plateH > 0 ? plateW / plateH : 16 / 9;
  const baseGate = computeGateRect(wrapSize.w, wrapSize.h, aspect);
  const gate = {
    w: baseGate.w * gateZoom,
    h: baseGate.h * gateZoom,
    x: wrapSize.w * 0.5 - baseGate.w * gateZoom * 0.5 + gatePan[0],
    y: wrapSize.h * 0.5 - baseGate.h * gateZoom * 0.5 + gatePan[1],
  };

  const isCameraMode = viewMode === "camera";
  const canPickSolvedPeople =
    workMode === "human" &&
    displayMode !== "off" &&
    (scene?.characters.length || 0) > 0 &&
    !(isCameraMode && (selection.mode === "picked" || manualDrawMode || wipeEnabled));
  const wipePx = Math.max(0, Math.min(wrapSize.w, wipeRatio * wrapSize.w));
  const canDrawManualBox =
    isCameraMode && workMode === "human" && !wipeEnabled && manualDrawMode && !!activeManualPersonId && !!onDrawManualBox;
  // A solved person is shown as a body, not as a box: the boxes left in the picture are the people who can still be
  // added.  (The box of a solved person comes back only while that person's manual track is being edited.)
  const solvedIdSet = useMemo(
    () => new Set((scene?.characters || []).map((c) => c.person_id)),
    [scene?.characters]
  );
  const isBoxDrawn = (p: DetectedPerson): boolean => {
    if (workMode === "object") return false;
    if (!(showBoxes || canDrawManualBox)) return false;
    if (!showBoxes && !p.is_manual) return false;
    if (solvedIdSet.has(p.id) && !(canDrawManualBox && p.is_manual && p.id === activeManualPersonId)) return false;
    return true;
  };

  const isLinearOcio = Boolean(
    plate &&
      plate.ocio_view === "aces_sdr" &&
      ![
        "sRGB Encoded Rec.709 (sRGB)",
        "Gamma 2.2 Encoded Rec.709",
        "Gamma 1.8 Encoded Rec.709",
        "Camera Rec.709",
        "Raw",
      ].includes(plate.resolved_colorspace || "")
  );

  const clientToPlateClamped = (clientX: number, clientY: number): [number, number] | null => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    const u = Math.max(0, Math.min(1, (mx - gate.x) / Math.max(gate.w, 1)));
    const v = Math.max(0, Math.min(1, (my - gate.y) / Math.max(gate.h, 1)));
    return [Math.round(u * plateW * 10) / 10, Math.round(v * plateH * 10) / 10];
  };

  const updateWipeFromClientX = (clientX: number) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || !onWipeRatioChange) return;
    const r = Math.max(0.0, Math.min(1.0, (clientX - rect.left) / rect.width));
    onWipeRatioChange(r);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!isCameraMode) return;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const nextZoom = Math.min(10.0, Math.max(0.25, gateZoom * factor));
    onGateChange(nextZoom, gatePan);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1) e.preventDefault();
    if (!isCameraMode) return;

    // Maya-style 2D camera gate navigation:
    // MMB (or Alt+LMB) = Pan, RMB (or Alt+RMB) = Dolly Zoom, LMB = Wipe drag (when wipeEnabled) or Pick
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      drag2dRef.current = {
        kind: "pan",
        startX: e.clientX,
        startY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
        pan0: [...gatePan] as [number, number],
        zoom0: gateZoom,
      };
    } else if (e.button === 2) {
      e.preventDefault();
      drag2dRef.current = {
        kind: "zoom",
        startX: e.clientX,
        startY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
        pan0: [...gatePan] as [number, number],
        zoom0: gateZoom,
      };
    } else if (e.button === 0) {
      if (wipeEnabled) {
        e.preventDefault();
        updateWipeFromClientX(e.clientX);
        drag2dRef.current = {
          kind: "wipe",
          startX: e.clientX,
          startY: e.clientY,
          lastX: e.clientX,
          lastY: e.clientY,
          pan0: [...gatePan] as [number, number],
          zoom0: gateZoom,
        };
      } else if (canDrawManualBox) {
        e.preventDefault();
        const pt0 = clientToPlateClamped(e.clientX, e.clientY) || [0, 0];
        drag2dRef.current = {
          kind: "draw_box",
          startX: e.clientX,
          startY: e.clientY,
          lastX: e.clientX,
          lastY: e.clientY,
          pan0: [...gatePan] as [number, number],
          zoom0: gateZoom,
          plateStart: pt0,
        };
        setLiveDrawBox([pt0[0], pt0[1], pt0[0], pt0[1]]);
      } else {
        drag2dRef.current = {
          kind: "click",
          startX: e.clientX,
          startY: e.clientY,
          lastX: e.clientX,
          lastY: e.clientY,
          pan0: [...gatePan] as [number, number],
          zoom0: gateZoom,
        };
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const st = drag2dRef.current;
    if (!st) return;
    if (st.kind === "wipe") {
      updateWipeFromClientX(e.clientX);
    } else if (st.kind === "draw_box" && st.plateStart) {
      const pt1 = clientToPlateClamped(e.clientX, e.clientY);
      if (pt1) {
        setLiveDrawBox([
          Math.min(st.plateStart[0], pt1[0]),
          Math.min(st.plateStart[1], pt1[1]),
          Math.max(st.plateStart[0], pt1[0]),
          Math.max(st.plateStart[1], pt1[1]),
        ]);
      }
    } else if (st.kind === "pan") {
      const dx = e.clientX - st.startX;
      const dy = e.clientY - st.startY;
      onGateChange(gateZoom, [st.pan0[0] + dx, st.pan0[1] + dy]);
    } else if (st.kind === "zoom") {
      const dx = e.clientX - st.startX;
      const dy = e.clientY - st.startY;
      const nextZoom = Math.min(10.0, Math.max(0.25, st.zoom0 * Math.exp((dx - dy) * 0.005)));
      onGateChange(nextZoom, gatePan);
    }
    st.lastX = e.clientX;
    st.lastY = e.clientY;
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const st = drag2dRef.current;
    drag2dRef.current = null;
    if (st && st.kind === "draw_box" && st.plateStart) {
      const pt1 = clientToPlateClamped(e.clientX, e.clientY) || st.plateStart;
      const x1 = Math.min(st.plateStart[0], pt1[0]);
      const y1 = Math.min(st.plateStart[1], pt1[1]);
      const x2 = Math.max(st.plateStart[0], pt1[0]);
      const y2 = Math.max(st.plateStart[1], pt1[1]);
      setLiveDrawBox(null);
      if (x2 - x1 >= 8 && y2 - y1 >= 12 && onDrawManualBox) {
        onDrawManualBox(frame, [x1, y1, x2, y2]);
      }
      return;
    }
    setLiveDrawBox(null);
    if (!st || st.kind !== "click" || !wrapRef.current) return;
    if (Math.hypot(e.clientX - st.startX, e.clientY - st.startY) > 5) return;

    const rect = wrapRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const u = (mx - gate.x) / Math.max(gate.w, 1);
    const v = (my - gate.y) / Math.max(gate.h, 1);
    const inGate = u >= 0 && u <= 1 && v >= 0 && v <= 1;
    const px = Math.round(u * plateW);
    const py = Math.round(v * plateH);
    if (workMode === "object" && onPickObjectPoint) {
      if (!inGate) return;
      const lbl: 0 | 1 = e.shiftKey ? 0 : objectClickLabel;
      onPickObjectPoint(frame, px, py, lbl);
      return;
    }
    if (selection.mode === "picked" && onPickPoint) {
      if (inGate) onPickPoint(frame, px, py);
      return;
    }
    // What can be clicked is what is drawn.  A BOX (somebody not solved yet) chooses who gets solved; a BODY
    // (somebody solved) chooses whether that person is exported.  A box under the cursor wins over a body, so
    // picking the next person to solve is never taken over by a solved neighbour standing in front.
    if (inGate && onToggleBoxPerson) {
      let bestId = -1;
      let bestArea = Infinity;
      for (const p of detectedPeople) {
        if (!isBoxDrawn(p)) continue;
        const b = p.boxes[String(frame)];
        if (!b) continue;
        if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]) {
          const area = (b[2] - b[0]) * (b[3] - b[1]);
          if (area < bestArea) {
            bestArea = area;
            bestId = p.id;
          }
        }
      }
      if (bestId > 0) {
        onToggleBoxPerson(bestId);
        return;
      }
    }
    if (canPickSolvedPeople && onToggleExportPerson) {
      const hitId = pickCharacterAt(pickBridgeRef.current, pickRegistry, e.clientX, e.clientY);
      if (hitId !== null) onToggleExportPerson(hitId);
    }
  };

  return (
    <div
      ref={wrapRef}
      className="viewport-canvas-wrap"
      onContextMenu={(e) => e.preventDefault()}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        drag2dRef.current = null;
        setLiveDrawBox(null);
      }}
      onPointerDown={(e) => {
        // perspective view: remember where the button went down, so a click (not an orbit drag) can pick a person
        click3dRef.current = !isCameraMode && e.button === 0 && !e.altKey ? { x: e.clientX, y: e.clientY } : null;
      }}
      onPointerUp={(e) => {
        const st3 = click3dRef.current;
        click3dRef.current = null;
        if (!st3 || isCameraMode || !canPickSolvedPeople || !onToggleExportPerson) return;
        if (Math.hypot(e.clientX - st3.x, e.clientY - st3.y) > 5) return;
        const hitId = pickCharacterAt(pickBridgeRef.current, pickRegistry, e.clientX, e.clientY);
        if (hitId !== null) onToggleExportPerson(hitId);
      }}
      style={{
        cursor:
          isCameraMode && wipeEnabled
            ? "ew-resize"
            : canDrawManualBox || (isCameraMode && (workMode === "object" || selection.mode === "picked"))
            ? "crosshair"
            : "default",
      }}
    >
      <Canvas
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        onCreated={({ gl }) => {
          gl.localClippingEnabled = true;
        }}
        camera={{ fov: 42, near: 2, far: 250000, position: [250, 180, 350] }}
      >
        <PickBridgeProbe bridgeRef={pickBridgeRef} />
        <ambientLight intensity={0.58} />
        <hemisphereLight args={["#f8fafc", "#334155", 0.55]} />
        <directionalLight position={[300, 600, 400]} intensity={1.15} />
        <directionalLight position={[-300, 220, -300]} intensity={0.5} />

        {activeLookCam ? (
          <LookThroughController
            cam={activeLookCam}
            frame={frame}
            plateW={plateW}
            plateH={plateH}
            shotId={shotId}
            plateRev={plateRev}
            hasPlate={!!plate}
            isLinearOcio={isLinearOcio}
            grade={grade}
            bypassGrade={bypassGrade}
            gateZoom={gateZoom}
            gatePan={gatePan}
          />
        ) : (
          <MayaPerspectiveController scene={scene} shotId={shotId} />
        )}

        <WipePlaneController wipePlane={wipePlane} wipeEnabled={wipeEnabled} wipeRatio={wipeRatio} />

        {viewMode === "persp" && (
          <>
            <Grid
              args={[4000, 4000]}
              cellSize={50}
              cellThickness={0.5}
              cellColor="#1e293b"
              sectionSize={200}
              sectionThickness={1.1}
              sectionColor="#334155"
              fadeDistance={3500}
              infiniteGrid
            />
            <axesHelper args={[80]} />
            {sceneCam && <CameraGizmo cam={sceneCam} frame={frame} />}
          </>
        )}

        {displayMode !== "off" && showSceneScan && scene?.scene_scan && (
          <SceneScanMeshView scan={scene.scene_scan} isCameraMode={isCameraMode} clipPlanes={clipPlanes} />
        )}

        {displayMode !== "off" &&
          scene?.characters.map((ch) => (
            <CharacterRigView
              key={ch.person_id}
              ch={ch}
              frame={frame}
              displayMode={displayMode}
              opacity={activeLookCam ? overlayOpacity : Math.max(overlayOpacity, 0.92)}
              autoHide={autoHide}
              clipPlanes={clipPlanes}
              dimmed={!!exportPersonIds && !exportPersonIds.has(ch.person_id)}
              pickRegistry={pickRegistry}
            />
          ))}

        {displayMode !== "off" &&
          (scene?.objects || []).map((obj) => (
            <RigidObjectMeshView
              key={obj.object_id}
              obj={obj}
              frame={frame}
              displayMode={displayMode}
              opacity={activeLookCam ? overlayOpacity : Math.max(overlayOpacity, 0.92)}
              autoHide={autoHide}
              clipPlanes={clipPlanes}
            />
          ))}
      </Canvas>

      {/* Film Gate Border + 2D Person Detection Boxes / Object Mask Contours Overlay in Camera Mode */}
      {isCameraMode && (plate || activeLookCam) && (
        <svg
          width={wrapSize.w}
          height={wrapSize.h}
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        >
          <rect
            x={gate.x}
            y={gate.y}
            width={gate.w}
            height={gate.h}
            fill="none"
            stroke="rgba(79, 141, 255, 0.45)"
            strokeWidth={1.2}
            strokeDasharray="6 4"
          />
          <text x={gate.x + 8} y={gate.y + 16} fill="rgba(226, 232, 240, 0.78)" fontSize={11}>
            {sceneCam && sceneCam.role === "aligned"
              ? `${sceneCam.name} (${plateW}×${plateH})`
              : scene?.characters.length || scene?.objects.length
              ? `原点标准摄影机 (${plateW}×${plateH})`
              : `${plate?.source_name || "序列画面"} (${plateW}×${plateH})`}
          </text>

          <g
            style={
              wipeEnabled
                ? { clipPath: `inset(0 ${Math.max(0, wrapSize.w - wipePx)}px 0 0)` }
                : undefined
            }
          >
          {(showBoxes || canDrawManualBox) &&
            workMode !== "object" &&
            detectedPeople.map((p) => {
              if (!isBoxDrawn(p)) return null;
              const box = p.boxes[String(frame)];
              if (!box) return null;
              const [x1, y1, x2, y2] = box;
              const sx = gate.x + (x1 / plateW) * gate.w;
              const sy = gate.y + (y1 / plateH) * gate.h;
              const sw = ((x2 - x1) / plateW) * gate.w;
              const sh = ((y2 - y1) / plateH) * gate.h;
              const chosen = selectedPersonIds.has(p.id);
              const col = personColor(p.id);
              const mSt = p.is_manual ? manualStatusesByPerson[p.id]?.[frame] : undefined;
              const isKf = mSt === "key_active";
              const isActMan = p.is_manual && p.id === activeManualPersonId;
              const badgeText = p.is_manual
                ? `${chosen ? "✓ " : ""}#${p.id} ${isKf ? "◆关键帧" : "↔插值"}`
                : chosen
                ? `✓ #${p.id}`
                : `#${p.id}`;
              const badgeW = p.is_manual ? (chosen ? 96 : 84) : chosen ? 58 : 42;
              return (
                <g key={p.id}>
                  <rect
                    x={sx}
                    y={sy}
                    width={Math.max(sw, 2)}
                    height={Math.max(sh, 2)}
                    fill={chosen ? `${col}1c` : "none"}
                    stroke={col}
                    strokeWidth={isActMan ? 2.6 : chosen ? 2.2 : 1.1}
                    strokeDasharray={
                      p.is_manual
                        ? isKf
                          ? undefined
                          : "7 3"
                        : chosen
                        ? undefined
                        : "4 3"
                    }
                  />
                  {p.is_manual && isKf && (
                    <>
                      <rect x={sx - 3} y={sy - 3} width={6} height={6} fill={col} stroke="#090d14" strokeWidth={1} />
                      <rect x={sx + Math.max(sw, 2) - 3} y={sy - 3} width={6} height={6} fill={col} stroke="#090d14" strokeWidth={1} />
                      <rect x={sx - 3} y={sy + Math.max(sh, 2) - 3} width={6} height={6} fill={col} stroke="#090d14" strokeWidth={1} />
                      <rect x={sx + Math.max(sw, 2) - 3} y={sy + Math.max(sh, 2) - 3} width={6} height={6} fill={col} stroke="#090d14" strokeWidth={1} />
                    </>
                  )}
                  <rect x={sx} y={Math.max(gate.y, sy - 18)} width={badgeW} height={17} rx={3} fill={col} />
                  <text
                    x={sx + 5}
                    y={Math.max(gate.y, sy - 18) + 12}
                    fill="#090d14"
                    fontSize={10.5}
                    fontWeight={700}
                  >
                    {badgeText}
                  </text>
                </g>
              );
            })}

          {/* Live Rubber-Band Rectangle While Dragging Manual Box */}
          {liveDrawBox && activeManualPersonId && (
            (() => {
              const [lx1, ly1, lx2, ly2] = liveDrawBox;
              const sx = gate.x + (lx1 / plateW) * gate.w;
              const sy = gate.y + (ly1 / plateH) * gate.h;
              const sw = Math.max(2, ((lx2 - lx1) / plateW) * gate.w);
              const sh = Math.max(2, ((ly2 - ly1) / plateH) * gate.h);
              const col = personColor(activeManualPersonId);
              return (
                <g>
                  <rect
                    x={sx}
                    y={sy}
                    width={sw}
                    height={sh}
                    fill={`${col}28`}
                    stroke="#facc15"
                    strokeWidth={2.4}
                    strokeDasharray="6 3"
                  />
                  <rect x={sx} y={Math.max(gate.y, sy - 20)} width={126} height={18} rx={3} fill="#facc15" />
                  <text
                    x={sx + 6}
                    y={Math.max(gate.y, sy - 20) + 12.5}
                    fill="#090d14"
                    fontSize={10.5}
                    fontWeight={700}
                  >
                    {`#${activeManualPersonId} 新关键帧 ${Math.round(lx2 - lx1)}×${Math.round(ly2 - ly1)}`}
                  </text>
                </g>
              );
            })()
          )}

          {showObjectMasks &&
            trackedObjects.map((tobj) => {
              const polys = tobj.contours?.[String(frame)];
              const box = tobj.boxes?.[String(frame)];
              if (!polys?.length && !box) return null;
              const col = objectColor(tobj.id);
              const isAct = tobj.id === activeObjectId;
              return (
                <g key={`obj-mask-${tobj.id}`}>
                  {(polys || []).map((poly, pidx) => {
                    const ptsStr = poly
                      .map(([px, py]) => `${gate.x + (px / plateW) * gate.w},${gate.y + (py / plateH) * gate.h}`)
                      .join(" ");
                    return (
                      <polygon
                        key={pidx}
                        points={ptsStr}
                        fill={`${col}32`}
                        stroke={col}
                        strokeWidth={isAct ? 2.2 : 1.5}
                      />
                    );
                  })}
                  {box && (
                    <>
                      <rect
                        x={gate.x + (box[0] / plateW) * gate.w}
                        y={Math.max(gate.y, gate.y + (box[1] / plateH) * gate.h - 18)}
                        width={62}
                        height={17}
                        rx={3}
                        fill={col}
                      />
                      <text
                        x={gate.x + (box[0] / plateW) * gate.w + 5}
                        y={Math.max(gate.y, gate.y + (box[1] / plateH) * gate.h - 18) + 12}
                        fill="#090d14"
                        fontSize={10.5}
                        fontWeight={700}
                      >
                        {`物体 #${tobj.id}`}
                      </text>
                    </>
                  )}
                </g>
              );
            })}

          {workMode === "object" &&
            (showObjectMasks || !(scene?.objects?.length)) &&
            objectPrompts.map((op) => {
              const col = objectColor(op.id);
              return (op.points || [])
                .filter((pt) => pt.frame === frame)
                .map((pt, idx) => {
                  const cx = gate.x + (pt.x / plateW) * gate.w;
                  const cy = gate.y + (pt.y / plateH) * gate.h;
                  const isPos = pt.label === 1;
                  return (
                    <g key={`opt-${op.id}-${idx}`}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={6.5}
                        fill={isPos ? "#10b981" : "#ef4444"}
                        stroke="#fff"
                        strokeWidth={1.6}
                      />
                      <circle cx={cx} cy={cy} r={9.5} fill="none" stroke={col} strokeWidth={1.3} />
                      <text
                        x={cx}
                        y={cy + 3.5}
                        textAnchor="middle"
                        fill="#fff"
                        fontSize={11}
                        fontWeight={800}
                      >
                        {isPos ? "+" : "−"}
                      </text>
                    </g>
                  );
                });
            })}

          {workMode !== "object" &&
            selection.mode === "picked" &&
            (selection.picks || [])
              .filter((pk) => pk[0] === frame)
              .map((pk, idx) => {
                const cx = gate.x + (pk[1] / plateW) * gate.w;
                const cy = gate.y + (pk[2] / plateH) * gate.h;
                return (
                  <g key={idx}>
                    <circle cx={cx} cy={cy} r={6} fill="#facc15" stroke="#000" strokeWidth={1.5} />
                  </g>
                );
              })}
          </g>
        </svg>
      )}

      {/* Interactive Split-Wipe Curtain Handle & Divider */}
      {wipeEnabled && (
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: Math.max(0, Math.min(wrapSize.w - 2, wipePx - 12)),
            width: 24,
            cursor: "ew-resize",
            zIndex: 18,
            touchAction: "none",
          }}
          onPointerDown={(e) => {
            e.stopPropagation();
            wipeDraggingRef.current = true;
            (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
            updateWipeFromClientX(e.clientX);
          }}
          onPointerMove={(e) => {
            if (!wipeDraggingRef.current) return;
            e.stopPropagation();
            updateWipeFromClientX(e.clientX);
          }}
          onPointerUp={(e) => {
            if (!wipeDraggingRef.current) return;
            e.stopPropagation();
            wipeDraggingRef.current = false;
            try {
              (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
            } catch {
              // ignore
            }
          }}
        >
          {/* Glowing vertical split line */}
          <div
            style={{
              position: "absolute",
              top: 0,
              bottom: 0,
              left: 11,
              width: 2,
              background: "var(--accent)",
              boxShadow: "0 0 8px var(--accent-line), 0 0 1px #ffffff",
              pointerEvents: "none",
            }}
          />
          {/* Center circular drag handle */}
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: 12,
              transform: "translate(-50%, -50%)",
              width: 28,
              height: 28,
              borderRadius: "50%",
              background: "rgba(12, 15, 20, 0.92)",
              border: "1.5px solid var(--accent)",
              boxShadow: "0 2px 8px rgba(0, 0, 0, 0.55)",
              color: "#e8ecf3",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M5.5 4.5 2.5 8l3 3.5M10.5 4.5l3 3.5-3 3.5" />
            </svg>
          </div>
          {/* Side indicator pills */}
          {wipePx > 78 && (
            <span
              style={{
                position: "absolute",
                top: 10,
                right: 18,
                background: "rgba(12, 15, 20, 0.8)",
                border: "1px solid var(--accent-line)",
                color: "var(--accent-text)",
                fontSize: 11,
                lineHeight: "18px",
                padding: "0 7px",
                borderRadius: 5,
                whiteSpace: "nowrap",
                pointerEvents: "none",
              }}
            >
              跟踪模型
            </span>
          )}
          {wrapSize.w - wipePx > 78 && (
            <span
              style={{
                position: "absolute",
                top: 10,
                left: 18,
                background: "rgba(12, 15, 20, 0.8)",
                border: "1px solid var(--border-strong)",
                color: "var(--text-main)",
                fontSize: 11,
                lineHeight: "18px",
                padding: "0 7px",
                borderRadius: 5,
                whiteSpace: "nowrap",
                pointerEvents: "none",
              }}
            >
              原片
            </span>
          )}
        </div>
      )}

      {/* Camera Mode Manual Box Status / Draw Mode Banner */}
      {isCameraMode && workMode === "human" && activeManualPersonId && (
        (() => {
          const curSt = manualStatusesByPerson[activeManualPersonId]?.[frame];
          const isOcc = curSt === "occluded";
          const isOff = curSt === "off";
          if (!canDrawManualBox && !isOcc && !isOff) return null;
          return (
            <div
              style={{
                position: "absolute",
                right: 12,
                top: 10,
                pointerEvents: "none",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-end",
                gap: 6,
                zIndex: 14,
              }}
            >
              {canDrawManualBox && (
                <span className="tag tag-info is-solid">
                  {`绘制模式（#${activeManualPersonId}）：按住左键拖拽绘制选框，自动创建关键帧并插值`}
                </span>
              )}
              {isOcc && (
                <span className="tag tag-warn is-solid">
                  {`手动 #${activeManualPersonId}：当前帧完全遮挡（暂停解算，后续帧绘制选框后恢复）`}
                </span>
              )}
              {isOff && (
                <span className="tag tag-err is-solid">
                  {`手动 #${activeManualPersonId}：自当前帧起选框失效`}
                </span>
              )}
            </div>
          );
        })()
      )}

      {/* Perspective Space Indicator Badge */}
      {viewMode === "persp" && scene && (
        <div
          style={{
            position: "absolute",
            left: 12,
            top: 10,
            pointerEvents: "none",
            display: "flex",
            gap: 8,
            alignItems: "center",
          }}
        >
          <span className={`tag is-solid ${sceneCam?.role === "aligned" ? "tag-info" : "tag-ok"}`}>
            {sceneCam?.role === "aligned"
              ? `已对齐摄影机空间：${sceneCam.name}（蓝色线为导入摄影机路径）`
              : "原点标准摄影机空间 (0, 0, 0)"}
          </span>
        </div>
      )}
    </div>
  );
};
