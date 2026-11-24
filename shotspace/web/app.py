"""FastAPI Web Server for ShotSpace."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import queue

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..config import get_paths
from ..io import ARCHIVE_EXTS, SEQUENCE_EXTS, OcioSettings, extract_sequence_archive, list_ocio_options
from ..server_config import ConfigStore
from ..solver import probe_torch_archs
from .workspace import WorkspaceManager


class CreateShotRequest(BaseModel):
    name: str = "新任务"


class AdminAssignRequest(BaseModel):
    ip: str = ""
    name: str = ""


class UpdateShotRequest(BaseModel):
    name: str | None = None
    work_mode: str | None = None
    ocio: dict | None = None
    grade: dict | None = None
    detect_params: dict | None = None
    manual_people: list[dict] | None = None
    selection: dict | None = None
    solve_params: dict | None = None
    active_person_ids: list[int] | None = None
    export_person_ids: list[int] | None = None
    object_prompts: list[dict] | None = None
    object_solve_params: dict | None = None
    active_object_ids: list[int] | None = None


class FinalizeSequenceRequest(BaseModel):
    input_colorspace: str = "auto"
    ocio_view: str = "aces_sdr"
    fps: float = 24.0


class UpdateOcioRequest(BaseModel):
    input_colorspace: str = "auto"
    ocio_view: str = "aces_sdr"


class CameraOptionsRequest(BaseModel):
    camera_path: str = ""
    unit: str = ""
    fbx_take: str = ""
    frame_offset: int = 0


class SceneScanOptionsRequest(BaseModel):
    unit: str = ""
    selected_paths: list[str] | None = None


class SceneScanAlignRequest(BaseModel):
    enabled: bool = True


class DetectRequest(BaseModel):
    threshold: float | None = None
    grade: dict | None = None


class SolveRequest(BaseModel):
    grade: dict | None = None
    detect_params: dict | None = None
    manual_people: list[dict] | None = None
    selection: dict | None = None
    solve_params: dict | None = None


class ObjectTrackRequest(BaseModel):
    object_prompts: list[dict] | None = None
    grade: dict | None = None


class ObjectSolveRequest(BaseModel):
    object_prompts: list[dict] | None = None
    object_solve_params: dict | None = None
    grade: dict | None = None


class RemoveSolvedPeopleRequest(BaseModel):
    # solved people to take out of the task; empty or missing = all of them
    person_ids: list[int] | None = None


class ExportRequest(BaseModel):
    formats: list[str] = ["usd_skel", "usd_cache", "fbx", "abc"]
    unit: str = ""
    fps: float | None = None
    prefix: str = ""


class AdminSetupRequest(BaseModel):
    port: int = 8765
    password: str
    cache_dir: str = ""


class AdminLoginRequest(BaseModel):
    password: str


class AdminPasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str


class AdminPortRequest(BaseModel):
    port: int


class AdminCacheDirRequest(BaseModel):
    cache_dir: str = ""


class AdminGpuToggleRequest(BaseModel):
    enabled: bool


class AdminRetentionRequest(BaseModel):
    retention_days: int = 30
    run_cleanup: bool = True


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "").strip()
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _clean_client_filename(raw: str | None, default: str = "") -> str:
    """Normalize uploaded/requested filenames from both Windows (`C:\\dir\\file.exr`) and POSIX (`dir/file.exr`) clients."""
    s = (raw or default).replace("\\", "/").strip()
    if ":" in s and len(s) >= 2 and s[1] == ":":
        s = s[2:]
    leaf = s.split("/")[-1].strip()
    return Path(leaf).name if leaf not in ("", ".", "..") else default


def create_app(workspaces_dir: Path | None = None, config_store: ConfigStore | None = None) -> FastAPI:
    paths = get_paths()
    cfg_store = config_store or ConfigStore()
    wm = WorkspaceManager(workspaces_dir=workspaces_dir, config_store=cfg_store)

    app = FastAPI(title="ShotSpace API", version=__version__)
    app.state.wm = wm
    app.state.cfg_store = cfg_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _viewer(request: Request) -> dict:
        """The computer a request comes from: its address and machine name (`machines.py`)."""
        return wm.machines.viewer(_client_ip(request))

    @app.middleware("http")
    async def only_own_tasks(request: Request, call_next):
        """Every call on one task (`/api/shots/<task>/...`) is answered only for the computer the task belongs to,
        or for a public task (v0.2.12).  The admin page has calls of its own and is not affected."""
        path = request.url.path
        if path.startswith("/api/shots/"):
            shot_id = path[len("/api/shots/"):].split("/", 1)[0]
            if shot_id:
                viewer = await run_in_threadpool(_viewer, request)
                allowed = await run_in_threadpool(wm.may_use, shot_id, viewer)
                if not allowed:
                    return JSONResponse({"detail": "这是另一台电脑的任务，本机不能打开"}, status_code=403)
        return await call_next(request)

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not cfg_store.verify_token(token):
            raise HTTPException(status_code=401, detail="管理员登录已过期或未授权，请重新输入后台密码")

    @app.get("/api/health")
    def api_health() -> dict:
        torch_ver, archs = probe_torch_archs(paths.sam3d_python)
        cfg = cfg_store.load()
        return {
            "status": "ok",
            "version": __version__,
            "initialized": cfg.initialized,
            "configured_port": cfg.port,
            "torch_version": torch_ver,
            "cuda_archs": list(archs or []),
            "ocio": list_ocio_options(),
            "queue": wm.scheduler.public_summary(),
            "toolchains": {
                "sam3d_ready": paths.sam3d_python.is_file() and (paths.sam3d_weights / "model.ckpt").is_file(),
                "vitdet_ready": (paths.vitdet_weights / "model_final_f05665.pkl").is_file(),
                "sam_obj_ready": paths.sam_obj_python.is_file()
                and (paths.sam31_weights / "sam3.1_multiplex.pt").is_file()
                and (paths.moge_weights / "model.pt").is_file()
                and (paths.sam3d_objects_weights / "checkpoints" / "pipeline.yaml").is_file()
                and (paths.sam3d_objects_weights / "checkpoints" / "ss_generator.ckpt").is_file(),
                "alembic_ready": paths.alembic_python.is_file(),
                "fbx_ready": paths.fbx_python.is_file(),
            },
        }

    @app.get("/api/ocio")
    def api_ocio_options() -> dict:
        return list_ocio_options()

    # ------------------------------------------------------------------ `/api/admin/*`

    @app.get("/api/admin/status")
    def api_admin_status(authorization: str | None = Header(default=None)) -> dict:
        cfg = cfg_store.load()
        token = authorization[7:].strip() if (authorization and authorization.lower().startswith("bearer ")) else ""
        return {
            "initialized": cfg.initialized,
            "authenticated": cfg_store.verify_token(token) if token else False,
            "port": cfg.port,
            "host": cfg.host,
            "default_cache_dir": str(paths.workspaces.resolve()),
        }

    @app.post("/api/admin/setup")
    def api_admin_first_setup(req: AdminSetupRequest) -> dict:
        cfg = cfg_store.load()
        if cfg.initialized:
            raise HTTPException(status_code=403, detail="系统已完成首次初始化，请使用后台密码登录")
        try:
            saved = cfg_store.initialize(port=req.port, password=req.password, cache_dir=req.cache_dir)
            if saved.cache_dir:
                wm.switch_cache_dir(saved.cache_dir)
            return {
                "ok": True,
                "port": saved.port,
                "cache_dir": str(wm.base_dir),
                "token": cfg_store.create_token(),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/login")
    def api_admin_login(req: AdminLoginRequest) -> dict:
        cfg = cfg_store.load()
        if not cfg.initialized:
            raise HTTPException(status_code=400, detail="系统尚未完成首次启动密码设置")
        if not cfg_store.verify_password(req.password):
            raise HTTPException(status_code=401, detail="后台密码错误")
        return {"ok": True, "token": cfg_store.create_token(), "port": cfg.port}

    @app.get("/api/admin/overview", dependencies=[Depends(require_admin)])
    def api_admin_overview() -> dict:
        cfg = cfg_store.load()
        data = wm.scheduler.overview()
        storage = wm.storage_stats()
        return {
            **data,
            "server": {
                "port": cfg.port,
                "host": cfg.host,
                "retention_days": cfg.retention_days,
                "updated_at": cfg.updated_at,
                **storage,
            },
        }

    @app.post("/api/admin/retention", dependencies=[Depends(require_admin)])
    def api_admin_set_retention(req: AdminRetentionRequest) -> dict:
        try:
            cfg = cfg_store.set_retention_days(req.retention_days)
            cleaned = wm.cleanup_expired() if req.run_cleanup else {"deleted_count": 0, "deleted_ids": []}
            return {
                "ok": True,
                "retention_days": cfg.retention_days,
                "cleaned": cleaned,
                "storage": wm.storage_stats(),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/gpus/all", dependencies=[Depends(require_admin)])
    def api_admin_toggle_all_gpus(req: AdminGpuToggleRequest) -> dict:
        return wm.scheduler.set_all_gpus_enabled(req.enabled)

    @app.post("/api/admin/gpus/{gpu_index}", dependencies=[Depends(require_admin)])
    def api_admin_toggle_gpu(gpu_index: int, req: AdminGpuToggleRequest) -> dict:
        try:
            return wm.scheduler.set_gpu_enabled(gpu_index, req.enabled)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/jobs/{job_id}/cancel", dependencies=[Depends(require_admin)])
    def api_admin_cancel_job(job_id: str) -> dict:
        ok = wm.scheduler.cancel_job(job_id)
        return {"ok": ok, "overview": wm.scheduler.overview()}

    @app.post("/api/admin/password", dependencies=[Depends(require_admin)])
    def api_admin_change_password(req: AdminPasswordChangeRequest) -> dict:
        try:
            cfg_store.change_password(req.old_password, req.new_password)
            return {"ok": True, "token": cfg_store.create_token()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/port", dependencies=[Depends(require_admin)])
    def api_admin_set_port(req: AdminPortRequest) -> dict:
        try:
            cfg = cfg_store.set_port(req.port)
            return {"ok": True, "port": cfg.port}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/cache_dir", dependencies=[Depends(require_admin)])
    def api_admin_set_cache_dir(req: AdminCacheDirRequest) -> dict:
        try:
            storage = wm.switch_cache_dir(req.cache_dir)
            return {"ok": True, "storage": storage}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------ Shots CRUD

    @app.get("/api/shots")
    def api_list_shots(request: Request) -> dict:
        viewer = _viewer(request)
        return {"shots": wm.list_shots(viewer), "queue": wm.scheduler.public_summary(), "machine": viewer}

    @app.post("/api/shots")
    def api_create_shot(req: CreateShotRequest, request: Request) -> dict:
        sw = wm.create_shot(name=req.name, owner=_viewer(request))
        return wm.enrich_shot_meta(sw.read_meta())

    @app.post("/api/shots/{shot_id}/claim")
    def api_claim_shot(shot_id: str, request: Request) -> dict:
        try:
            return wm.claim_shot(shot_id, _viewer(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/api/admin/shots", dependencies=[Depends(require_admin)])
    def api_admin_shots() -> dict:
        return {"shots": wm.all_tasks_with_owner()}

    @app.get("/api/admin/machines", dependencies=[Depends(require_admin)])
    def api_admin_machines() -> dict:
        return {"machines": wm.machines_overview()}

    @app.post("/api/admin/shots/{shot_id}/unclassified", dependencies=[Depends(require_admin)])
    def api_admin_make_unclassified(shot_id: str) -> dict:
        try:
            return wm.make_public(shot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/admin/shots/{shot_id}/owner", dependencies=[Depends(require_admin)])
    def api_admin_assign(shot_id: str, req: AdminAssignRequest) -> dict:
        try:
            return wm.assign_shot(shot_id, req.ip, req.name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/admin/shots/{shot_id}", dependencies=[Depends(require_admin)])
    def api_admin_delete_shot(shot_id: str) -> dict:
        try:
            return wm.admin_delete_shot(shot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}")
    def api_get_shot(shot_id: str) -> dict:
        try:
            return wm.enrich_shot_meta(wm.get_shot(shot_id).read_meta())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/shots/{shot_id}")
    def api_update_shot(shot_id: str, req: UpdateShotRequest) -> dict:
        try:
            return wm.update_settings(shot_id, req.model_dump(exclude_none=True))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/shots/{shot_id}")
    def api_delete_shot(shot_id: str, request: Request) -> dict:
        try:
            wm.delete_shot(shot_id, viewer=_viewer(request))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------------------------ Sequence Upload & OCIO Conversion

    @app.post("/api/shots/{shot_id}/sequence/begin")
    def api_sequence_begin(shot_id: str) -> dict:
        try:
            wm.begin_sequence_upload(shot_id)
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/sequence/frame")
    async def api_sequence_upload_frame(
        shot_id: str,
        request: Request,
        filename: str = Query(...),
    ) -> dict:
        try:
            seq_dir = wm.sequence_upload_dir(shot_id)
            safe_name = _clean_client_filename(filename)
            if not safe_name or safe_name.startswith(".") or safe_name.lower() in ("thumbs.db", "desktop.ini"):
                raise ValueError("无效文件名")
            ext = Path(safe_name).suffix.lower()
            is_archive = ext in ARCHIVE_EXTS or safe_name.lower().endswith(".tar.gz")
            if ext not in SEQUENCE_EXTS and not is_archive:
                raise ValueError(
                    f"不支持的文件格式 {ext}（支持序列图 {', '.join(sorted(SEQUENCE_EXTS))} 或 .zip/.tar 序列包）"
                )
            dest = seq_dir / safe_name
            with dest.open("wb") as f_out:
                async for chunk in request.stream():
                    f_out.write(chunk)
            if is_archive:
                extracted = extract_sequence_archive(dest, seq_dir)
                dest.unlink(missing_ok=True)
                return {"ok": True, "extracted": extracted}
            return {"ok": True, "filename": safe_name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/sequence/finalize")
    def api_sequence_finalize(shot_id: str, req: FinalizeSequenceRequest) -> dict:
        try:
            ocio = OcioSettings(input_colorspace=req.input_colorspace, ocio_view=req.ocio_view)
            return wm.finalize_sequence_upload(shot_id, ocio=ocio, fps=req.fps)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/ocio")
    def api_update_ocio(shot_id: str, req: UpdateOcioRequest) -> dict:
        try:
            ocio = OcioSettings(input_colorspace=req.input_colorspace, ocio_view=req.ocio_view)
            return wm.update_ocio_settings(shot_id, ocio)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/frames/{frame}")
    def api_get_frame(shot_id: str, frame: int) -> FileResponse:
        try:
            fpath = wm.get_preview_frame_path(shot_id, frame)
            return FileResponse(fpath, headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/thumbnail")
    def api_get_shot_thumbnail(shot_id: str) -> FileResponse:
        try:
            tpath = wm.get_shot_thumbnail_path(shot_id)
            return FileResponse(tpath, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/person_thumb/{person_id}")
    def api_get_person_thumbnail(shot_id: str, person_id: int) -> FileResponse:
        try:
            tpath = wm.get_person_thumbnail_path(shot_id, person_id)
            return FileResponse(tpath, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ------------------------------------------------------------------ Single Camera Upload (Y-up ONLY)

    @app.post("/api/shots/{shot_id}/camera/upload")
    async def api_upload_camera(
        shot_id: str,
        request: Request,
        filename: str = Query(default="camera.abc"),
        camera_path: str = Query(default=""),
        unit: str = Query(default="m"),
        fbx_take: str = Query(default=""),
        frame_offset: int = Query(default=0),
    ) -> dict:
        try:
            sw = wm.get_shot(shot_id)
            cam_dir = sw.root / "cameras"
            cam_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _clean_client_filename(filename, "camera.abc")
            dest = cam_dir / safe_name
            with dest.open("wb") as f_out:
                async for chunk in request.stream():
                    f_out.write(chunk)
            return wm.set_camera(
                shot_id,
                file_path=dest,
                camera_path=camera_path,
                unit=unit,
                fbx_take=fbx_take,
                frame_offset=frame_offset,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/camera/options")
    def api_camera_options(shot_id: str, req: CameraOptionsRequest) -> dict:
        try:
            return wm.update_uploaded_camera_options(
                shot_id,
                camera_path=req.camera_path,
                unit=req.unit,
                fbx_take=req.fbx_take,
                frame_offset=req.frame_offset,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/shots/{shot_id}/camera")
    def api_delete_camera(shot_id: str) -> dict:
        try:
            return wm.clear_camera(shot_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------ Separate Static Scene Scan Upload (.abc / .fbx / .usd)

    @app.post("/api/shots/{shot_id}/scene_scan/upload")
    async def api_upload_scene_scan(
        shot_id: str,
        request: Request,
        filename: str = Query(default="scene_scan.abc"),
        unit: str = Query(default="m"),
    ) -> dict:
        try:
            sw = wm.get_shot(shot_id)
            scan_dir = sw.root / "scene_scan"
            scan_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _clean_client_filename(filename, "scene_scan.abc")
            dest = scan_dir / safe_name
            with dest.open("wb") as f_out:
                async for chunk in request.stream():
                    f_out.write(chunk)
            return wm.set_scene_scan(shot_id, file_path=dest, unit=unit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/scene_scan/options")
    def api_scene_scan_options(shot_id: str, req: SceneScanOptionsRequest) -> dict:
        try:
            return wm.update_uploaded_scene_scan_options(
                shot_id, unit=req.unit, selected_paths=req.selected_paths
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/scene_scan/align")
    def api_scene_scan_align(shot_id: str, req: SceneScanAlignRequest, request: Request) -> dict:
        """Step 4: put the solved people on the scan (`enabled`), or show the task as it is without it."""
        try:
            return wm.set_scan_alignment(shot_id, bool(req.enabled), client_ip=_client_ip(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/shots/{shot_id}/scene_scan")
    def api_delete_scene_scan(shot_id: str) -> dict:
        try:
            return wm.clear_scene_scan(shot_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------ Queue Submission & SSE

    @app.post("/api/shots/{shot_id}/detect")
    def api_run_detect(shot_id: str, req: DetectRequest, request: Request) -> dict:
        try:
            return wm.enqueue_detect(
                shot_id,
                threshold=req.threshold,
                grade=req.grade,
                client_ip=_client_ip(request),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/solve")
    def api_run_solve(shot_id: str, req: SolveRequest, request: Request) -> dict:
        try:
            patch = req.model_dump(exclude_none=True)
            return wm.enqueue_solve(shot_id, params_patch=patch, client_ip=_client_ip(request))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/solved_people/remove")
    def api_remove_solved_people(shot_id: str, req: RemoveSolvedPeopleRequest) -> dict:
        try:
            return wm.remove_solved_people(shot_id, person_ids=req.person_ids)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/objects/track")
    def api_run_object_track(shot_id: str, req: ObjectTrackRequest, request: Request) -> dict:
        try:
            return wm.enqueue_object_track(
                shot_id,
                object_prompts=req.object_prompts,
                grade=req.grade,
                client_ip=_client_ip(request),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/objects/solve")
    def api_run_object_solve(shot_id: str, req: ObjectSolveRequest, request: Request) -> dict:
        try:
            patch = req.model_dump(exclude_none=True)
            return wm.enqueue_object_solve(shot_id, params_patch=patch, client_ip=_client_ip(request))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/cancel")
    def api_cancel_shot_task(shot_id: str) -> dict:
        try:
            return wm.cancel_shot_task(shot_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/events")
    async def api_shot_events(shot_id: str, request: Request) -> StreamingResponse:
        try:
            sw = wm.get_shot(shot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        q: queue.Queue[dict] = queue.Queue()

        def _listener(ev: dict) -> None:
            q.put(ev)

        sw.subscribers.append(_listener)

        async def _gen():
            try:
                yield f"data: {json.dumps({'event': 'connected', 'meta': wm.enrich_shot_meta(sw.read_meta())}, ensure_ascii=False)}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.to_thread(q.get, True, 1.0)
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                if _listener in sw.subscribers:
                    sw.subscribers.remove(_listener)

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ 3D Viewport Data & Direct Format Downloads

    @app.get("/api/shots/{shot_id}/view/manifest")
    def api_view_manifest(shot_id: str) -> JSONResponse:
        try:
            sw = wm.get_shot(shot_id)
            mf = sw.root / "aligned" / "view_manifest.json"
            if not mf.is_file():
                raise HTTPException(status_code=404, detail="尚未生成三维预览数据")
            return JSONResponse(
                content=json.loads(mf.read_text(encoding="utf-8")),
                headers={"Cache-Control": "no-cache"},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/view/scene.bin")
    def api_view_binary(shot_id: str, request: Request) -> Response:
        try:
            sw = wm.get_shot(shot_id)
            bf = sw.root / "aligned" / "view_scene.bin.gz"
            if not bf.is_file():
                raise HTTPException(status_code=404, detail="尚未生成三维二进制数据")
            accept_enc = request.headers.get("accept-encoding", "")
            gz_bytes = bf.read_bytes()
            if "gzip" in accept_enc.lower():
                return Response(
                    content=gz_bytes,
                    media_type="application/octet-stream",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
                )
            import gzip

            return Response(
                content=gzip.decompress(gz_bytes),
                media_type="application/octet-stream",
                headers={"Cache-Control": "no-cache"},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/download/{fmt}")
    def api_direct_download(shot_id: str, fmt: str, unit: str = Query(default="")) -> FileResponse:
        try:
            out_file = wm.export_single_path(shot_id, fmt=fmt, unit=unit)
            return FileResponse(out_file, filename=out_file.name, media_type="application/octet-stream")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/shots/{shot_id}/export")
    def api_export_shot(shot_id: str, req: ExportRequest) -> dict:
        try:
            produced = wm.export_deliverables(
                shot_id,
                formats=req.formats,
                unit=req.unit,
                fps=req.fps,
                prefix=req.prefix,
            )
            return {"exports": produced, "shot": wm.enrich_shot_meta(wm.get_shot(shot_id).read_meta())}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/shots/{shot_id}/exports/{filename}")
    def api_download_export(shot_id: str, filename: str) -> FileResponse:
        try:
            sw = wm.get_shot(shot_id)
            safe = _clean_client_filename(filename)
            f = sw.root / "exports" / safe
            if not f.is_file():
                raise HTTPException(status_code=404, detail=f"未找到导出文件：{safe}")
            return FileResponse(f, filename=safe, media_type="application/octet-stream")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ------------------------------------------------------------------ Static Frontend SPA (`/` and `/admin`)

    dist = paths.webui_dist
    if dist.is_dir() and (dist / "index.html").is_file():
        assets_dir = dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        def serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            cand = (dist / full_path).resolve()
            if full_path and cand.is_file() and str(cand).startswith(str(dist.resolve())):
                return FileResponse(cand)
            return FileResponse(dist / "index.html")

    return app
