"""ShotSpace CLI (`shotspace serve`, `shotspace solve`, `shotspace passwd`, `shotspace doctor`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import DEFAULT_PORT, get_paths
from .server_config import ConfigStore
from .solver import list_gpus, probe_torch_archs

cli_app = typer.Typer(help="ShotSpace — 人体动作解算与摄影机空间转换工具", add_completion=False)
console = Console()


@cli_app.command("serve")
def cmd_serve(
    host: Optional[str] = typer.Option(None, "--host", help="监听地址（默认 0.0.0.0）"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="监听端口（首次启动将保存至配置文件）"),
    admin_password: Optional[str] = typer.Option(None, "--admin-password", help="首次启动时非交互式设置 /admin 后台密码"),
    workspaces: Optional[Path] = typer.Option(None, "--workspaces", help="工作区目录路径"),
) -> None:
    """启动 ShotSpace Web 服务与后台队列（首次启动引导设置端口与 /admin 密码）。"""
    import uvicorn

    from .web.app import create_app

    store = ConfigStore()
    cfg = store.load()

    if not cfg.initialized:
        if admin_password:
            use_port = int(port if port is not None else DEFAULT_PORT)
            use_host = host or "0.0.0.0"
            cfg = store.initialize(port=use_port, password=admin_password, host=use_host)
            console.print(f"[green]✓ 已完成首次初始化配置：[/green]端口 [bold]{cfg.port}[/bold]，后台入口 [bold]/admin[/bold]")
        elif sys.stdin.isatty():
            console.print("\n[bold cyan]=== ShotSpace 首次启动设置 ===[/bold cyan]")
            chosen_port = int(
                port
                if port is not None
                else typer.prompt("请设置 Web 服务监听端口", default=DEFAULT_PORT, type=int)
            )
            while True:
                pwd1 = typer.prompt("请设置 /admin 后台管理密码（至少 4 位）", hide_input=True)
                pwd2 = typer.prompt("请再次输入确认后台密码", hide_input=True)
                if pwd1 != pwd2:
                    console.print("[red]两次输入的密码不一致，请重试。[/red]")
                    continue
                if len(pwd1) < 4:
                    console.print("[red]密码长度不能少于 4 位，请重试。[/red]")
                    continue
                break
            cfg = store.initialize(port=chosen_port, password=pwd1, host=host or "0.0.0.0")
            console.print(f"[green]✓ 配置已保存至 {store.path}[/green]\n")
        else:
            console.print(
                "[yellow]提示：尚未初始化后台密码，启动后可访问 /admin 页面完成首次端口与密码设置。[/yellow]"
            )

    bind_host = host or cfg.host or "0.0.0.0"
    bind_port = int(port if port is not None else (cfg.port or DEFAULT_PORT))
    if cfg.initialized and port is not None and port != cfg.port:
        store.set_port(bind_port)

    console.print(
        f"[bold cyan]ShotSpace v{__version__}[/bold cyan] 运行中 -> "
        f"工作台 [green]http://{bind_host}:{bind_port}/[/green] · "
        f"后台 [green]http://{bind_host}:{bind_port}/admin[/green]"
    )
    app = create_app(workspaces_dir=workspaces, config_store=store)
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")


@cli_app.command("init")
def cmd_init(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址（默认 0.0.0.0）"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="首次启动监听端口"),
    admin_password: Optional[str] = typer.Option(None, "--admin-password", help="首次启动后台管理密码"),
    force: bool = typer.Option(False, "--force", help="即使已初始化也重新配置端口与后台密码"),
) -> None:
    """首次运行交互式配置监听端口与 `/admin` 后台管理密码。"""
    store = ConfigStore()
    cfg = store.load()
    if cfg.initialized and cfg.password_hash and not force:
        return

    console.print("\n[bold cyan]========================================================================[/bold cyan]")
    console.print("[bold cyan]  入镜 · ShotSpace 首次运行初始化配置（端口与 /admin 后台密码）[/bold cyan]")
    console.print("[bold cyan]========================================================================[/bold cyan]")

    if admin_password:
        chosen_port = int(port if port is not None else DEFAULT_PORT)
        saved = store.initialize(port=chosen_port, password=admin_password, host=host or "0.0.0.0")
        console.print(f"[green]✓ 首次运行配置已保存：[/green]端口 [bold]{saved.port}[/bold] · 配置文件 {store.path}\n")
        return

    if not sys.stdin.isatty():
        console.print("[red]✕ 首次运行必须配置监听端口与 /admin 后台管理密码，请在交互式终端执行启动脚本！[/red]")
        raise typer.Exit(code=1)

    default_p = int(port if port is not None else DEFAULT_PORT)
    while True:
        chosen_port = typer.prompt("请输入 Web 服务监听端口", default=default_p, type=int)
        if 1 <= chosen_port <= 65535:
            break
        console.print("[red]端口号必须在 1–65535 之间，请重新输入。[/red]")

    while True:
        pwd1 = typer.prompt("请设置 /admin 后台管理密码（至少 4 位）", hide_input=True)
        pwd2 = typer.prompt("请再次输入确认后台管理密码", hide_input=True)
        if pwd1 != pwd2:
            console.print("[red]两次输入的密码不一致，请重试。[/red]")
            continue
        if len(pwd1) < 4:
            console.print("[red]密码长度不能少于 4 位，请重试。[/red]")
            continue
        break

    saved = store.initialize(port=chosen_port, password=pwd1, host=host or "0.0.0.0")
    console.print(f"[green]✓ 配置已加密保存至 {store.path}（端口: {saved.port}）[/green]\n")


@cli_app.command("passwd")
def cmd_passwd(
    password: Optional[str] = typer.Option(None, "--password", help="直接指定新后台密码"),
) -> None:
    """重置或修改 `/admin` 后台管理密码。"""
    store = ConfigStore()
    new_pwd = password or typer.prompt("请输入新的 /admin 后台管理密码", hide_input=True, confirmation_prompt=True)
    store.force_set_password(new_pwd)
    console.print(f"[green]✓ 后台管理密码已更新（配置文件：{store.path}）[/green]")


@cli_app.command("doctor")
def cmd_doctor() -> None:
    """自检运行环境、CUDA 架构、权重与三维格式工具链。"""
    paths = get_paths()
    store = ConfigStore()
    cfg = store.load()
    table = Table(title=f"ShotSpace v{__version__} 环境自检")
    table.add_column("组件", style="cyan")
    table.add_column("状态", style="bold")
    table.add_column("详情")

    checks = [
        ("SAM 3D Body Python", paths.sam3d_python.is_file(), str(paths.sam3d_python)),
        ("SAM 3D Body 权重 (DINOv3+MHR)", (paths.sam3d_weights / "model.ckpt").is_file(), str(paths.sam3d_weights)),
        ("DINOv3 本地离线仓库", paths.dinov3_repo.is_dir(), str(paths.dinov3_repo)),
        ("SAM 3.1 人物跟踪环境", paths.sam_obj_python.is_file(), str(paths.sam_obj_python)),
        ("SAM 3.1 视频分割跟踪权重", (paths.sam31_weights / "sam3.1_multiplex.pt").is_file(), str(paths.sam31_weights)),
        ("MoGe 权重（物体匹配，可选）", (paths.moge_weights / "model.pt").is_file(), str(paths.moge_weights)),
        (
            "3D 物体重建权重（物体匹配，可选）",
            (paths.sam3d_objects_weights / "checkpoints" / "pipeline.yaml").is_file()
            or (paths.triposr_weights / "model.ckpt").is_file(),
            str(paths.sam3d_objects_weights if (paths.sam3d_objects_weights / "checkpoints" / "pipeline.yaml").is_file() else paths.triposr_weights),
        ),
        ("PyAlembic (.abc) 环境", paths.alembic_python.is_file(), str(paths.alembic_python)),
        ("Autodesk FBX SDK (.fbx) 环境", paths.fbx_python.is_file(), str(paths.fbx_python)),
        ("WebUI 前端静态包", (paths.webui_dist / "index.html").is_file(), str(paths.webui_dist)),
        ("后台配置 (/admin)", cfg.initialized, f"port={cfg.port}, disabled_gpus={cfg.disabled_gpus}"),
    ]
    for label, ok, detail in checks:
        table.add_row(label, "[green]OK[/green]" if ok else "[yellow]PENDING[/yellow]", detail)

    torch_ver, archs = probe_torch_archs(paths.sam3d_python)
    table.add_row(
        "PyTorch + CUDA 架构",
        "[green]OK[/green]" if archs else "[yellow]WARN[/yellow]",
        f"{torch_ver} ({', '.join(archs or [])})",
    )

    for g in list_gpus(paths.sam3d_python):
        enabled = g.index not in cfg.disabled_gpus
        st = (
            "[green]ENABLED[/green]"
            if (g.compatible and enabled)
            else ("[yellow]DISABLED[/yellow]" if g.compatible else "[red]INCOMPATIBLE[/red]")
        )
        table.add_row(
            f"GPU {g.index}: {g.name}",
            st,
            f"CC {g.compute_cap} ({g.sm}) · {g.memory_free_mb}/{g.memory_total_mb} MB Free",
        )

    console.print(table)


@cli_app.command("solve")
def cmd_solve(
    sequence: Path = typer.Option(..., "--sequence", "--plate", help="去畸变序列图目录 (.exr/.dpx/.tif/.png/.jpg)"),
    camera: Path = typer.Option(..., "--camera", help="跟踪摄影机文件 (.abc/.fbx/.usd，必填)"),
    scan: Optional[Path] = typer.Option(None, "--scan", help="静态场景扫描模型文件 (.abc/.fbx/.usd，可选)"),
    colorspace: str = typer.Option("auto", "--colorspace", help="OCIO 输入色彩空间（默认 auto: .exr=ACEScg）"),
    ocio_view: str = typer.Option("aces_sdr", "--ocio-view", help="OCIO 转换模式: aces_sdr | un_tone_mapped"),
    name: str = typer.Option("CLI 任务", "--name", help="任务名称"),
    exposure: float = typer.Option(0.0, "--exposure", help="曝光补偿 EV [-4, +4]"),
    brightness: float = typer.Option(0.0, "--brightness", help="亮度偏移 [-0.5, +0.5]"),
    gamma: float = typer.Option(1.0, "--gamma", help="Gamma [0.2, 3.0]"),
    select_mode: str = typer.Option("largest", "--select", help="选人规则: largest | top | all | ids"),
    use_mask: bool = typer.Option(False, "--use-mask/--no-use-mask", help="可选开启 SAM 3.1 人物轮廓 Mask 引导"),
    smoothing: float = typer.Option(0.5, "--smoothing", help="平滑强度 [0..1]"),
    formats: str = typer.Option("usd_skel,usd_cache,fbx,abc", "--formats", help="导出格式列表，逗号分隔"),
    unit: str = typer.Option("m", "--unit", help="摄影机与导出单位: m | cm"),
) -> None:
    """命令行一键执行序列图加载（OCIO ACEScg→sRGB）、摄影机导入、三维解算与 4 格式导出。"""
    from .io import OcioSettings
    from .web.workspace import WorkspaceManager

    wm = WorkspaceManager()
    sw = wm.create_shot(name=name)
    console.print(f"[cyan]已创建任务：[/cyan]{sw.shot_id} ({name})")
    wm.set_sequence_dir(sw.shot_id, sequence, ocio=OcioSettings(colorspace, ocio_view))
    wm.set_camera(sw.shot_id, camera, unit=unit)
    if scan is not None:
        wm.set_scene_scan(sw.shot_id, scan, unit=unit)
    wm.enqueue_solve(
        sw.shot_id,
        params_patch={
            "grade": {"exposure": exposure, "brightness": brightness, "gamma": gamma},
            "selection": {"mode": select_mode},
            "solve_params": {
                "smoothing": smoothing,
                "hand_refine": True,
                "use_mask": use_mask,
                "lock_shape": True,
            },
        },
        wait=True,
    )
    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    exps = wm.export_deliverables(sw.shot_id, formats=fmt_list, unit=unit)
    for e in exps:
        console.print(f"[green]✓ 已导出：[/green]{sw.root / 'exports' / e['filename']} ({e['size_bytes'] // 1024} KB)")


def main() -> None:
    if os.name == "nt":  # a Chinese Windows console / redirected log is GBK: never let a symbol stop the service
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    cli_app()


if __name__ == "__main__":
    main()
