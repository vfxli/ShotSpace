<div align="center">

# ShotSpace · 入镜

**Put the people of a live-action shot into the 3D space of its tracked camera**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-informational)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%2024GB-76B900)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)

[简体中文](README.md) | **English**

<img src="docs/images/camera_view.jpg" alt="Camera view" width="100%">

</div>

## Overview

Give ShotSpace an undistorted image sequence, the camera your matchmove already tracked and, optionally, a scan of
the set. It finds every person in the shot, estimates their body on every frame and solves each person over the
whole shot, delivering 3D people that line up exactly with the original camera space.

- **Exact registration**: the tracked camera's per-frame lens and position are used as they are.
- **Whole-shot solve**: one body shape per person, position constrained by every frame and the camera; with a scan, people stand on the real ground.
- **People and crowds**: SAM 3.1 tracks everyone, SAM 3D Body estimates every frame; solve people one by one or all at once.
- **Standard deliverables**: USD skeleton, USD point cache, FBX, Alembic, in metres.
- **Built for teams**: one GPU server, everyone works in Chrome; each computer sees its own tasks, an admin page manages all.

<img src="docs/images/perspective_view.jpg" alt="Perspective view" width="100%">

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (glibc ≥ 2.28) or Windows 10 / 11 x64 |
| GPU | NVIDIA, about 24 GB, driver ≥ 570 |
| Disk | about 30 GB (about 75 GB with object matching) |
| Browser | Chrome |

## Install

```bash
git clone https://github.com/vfxli/ShotSpace.git
cd ShotSpace
./setup.sh        # Linux
setup.bat         # Windows
```

The setup wizard asks for download sources, components and where the models come from, then installs everything
inside this folder without touching the system. To uninstall, delete the folder.

- **Mirrors**: PyPI, PyTorch, Python, source code and models each have several sources; the fastest is picked, failures fall back, downloads resume, and every file is checked by SHA-256.
- **Model weights**: from [ModelScope](https://modelscope.cn) (no account), HuggingFace (request access to [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) and [SAM 3.1](https://huggingface.co/facebook/sam3.1)), or a local folder.
- **Unattended**: `./setup.sh --yes --region global --weights huggingface`

## Usage

```bash
./shotspace.sh start      # Windows: shotspace.bat start
```

Open `http://<server>:8765` in Chrome; the admin page is `/admin`.

1. Create a task, load the image sequence and grade it
2. Load the tracked camera (`.abc` / `.fbx` / `.usd`)
3. Pick people and solve
4. Optional: load a set scan and align to it
5. Export USD / FBX / ABC

## How it works

| Stage | Method | Code |
|---|---|---|
| Find and track | SAM 3.1, forward and backward, one outline per person per frame | `solver/workers/people_tracker.py` |
| Per-frame body | SAM 3D Body with the camera's real intrinsics | `solver/workers/sam3d_worker.py` |
| Whole-shot solve | fixed body shape + camera + ground constraints | `solver/trajectory.py` · `solver/workers/body_solve.py` |

## License

ShotSpace is released under the [MIT License](LICENSE). The models downloaded at setup keep their own licences (SAM
License, DINOv3 License, ...); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
