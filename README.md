<div align="center">

# 入镜 · ShotSpace

**把实拍镜头里的人，解算到跟踪摄影机的三维空间中**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/平台-Linux%20%7C%20Windows-informational)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%2024GB-76B900)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
[![Powered by SAM 3.1](https://img.shields.io/badge/Powered%20by-SAM%203.1-0467DF)](https://github.com/facebookresearch/sam3)
[![Powered by SAM 3D Body](https://img.shields.io/badge/Powered%20by-SAM%203D%20Body-0467DF)](https://github.com/facebookresearch/sam-3d-body)

**简体中文** | [English](README.en.md)

<img src="docs/images/camera_view.jpg" alt="摄影机视角" width="100%">

</div>

## 简介

入镜是一个面向影视匹配移动（matchmove）的人物三维解算工具。它把 Meta 开源的 SAM 3.1 与 SAM 3D Body 等模型接入影视流程：输入去畸变序列、已跟踪好的摄影机和（可选）场景扫描，得到与镜头空间对齐的三维人物。

人物的识别、跟踪和逐帧三维估计都由这些模型完成；入镜负责把它们放进精确的摄影机空间，对整段镜头做统一解算，并输出常用的影视交付格式。

- 直接使用跟踪摄影机的逐帧内外参
- 整段镜头统一解算，有场景扫描时利用地面约束
- 支持多人，可逐个累加解算
- 导出 USD 骨架、USD 点缓存、FBX、Alembic
- 网页界面，多台电脑共用一台显卡服务器

<img src="docs/images/perspective_view.jpg" alt="透视图" width="100%">

## 环境要求

| | |
|---|---|
| 系统 | Linux x86_64（glibc ≥ 2.28）或 Windows 10 / 11 x64 |
| 显卡 | NVIDIA，约 24 GB 显存，驱动 ≥ 570 |
| 磁盘 | 约 30 GB（含物体匹配约 75 GB） |
| 浏览器 | Chrome |

## 安装

```bash
git clone https://github.com/vfxli/ShotSpace.git
cd ShotSpace
./setup.sh        # Linux
setup.bat         # Windows
```

安装向导会依次询问下载源、安装内容和模型来源，所有内容装在本文件夹内，不修改系统环境；卸载即删除文件夹。

- **多镜像源**：PyPI、PyTorch、Python、源码、模型均有国内外多个下载源，自动测速、失败切换、断点续传，全部按 SHA-256 校验。
- **模型权重**：默认从 [ModelScope](https://modelscope.cn) 下载（无需账号）；也可选 HuggingFace（需申请 [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) 与 [SAM 3.1](https://huggingface.co/facebook/sam3.1) 访问权限）或本地文件夹。
- **静默安装**：`./setup.sh --yes --region cn --weights modelscope`

## 使用

```bash
./shotspace.sh start      # Windows: shotspace.bat start
```

浏览器打开 `http://<服务器地址>:8765`，管理后台为 `/admin`。

1. 新建任务，导入序列图并调色
2. 导入跟踪摄影机（`.abc` / `.fbx` / `.usd`）
3. 选择人物，解算
4. 可选：导入场景扫描，按扫描对齐
5. 导出 USD / FBX / ABC

## 基于以下开源项目

入镜站在这些项目的肩膀上，感谢它们的作者：

| 项目 | 用途 |
|---|---|
| [SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3)（Meta） | 画面中人物的识别与跟踪 |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)（Meta） | 逐帧人体姿态与体型估计，使用 MHR 人体模型 |
| [DINOv3](https://github.com/facebookresearch/dinov3)（Meta） | SAM 3D Body 的图像特征主干 |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)、[MoGe](https://github.com/microsoft/MoGe)、[TripoSR](https://github.com/VAST-AI-Research/TripoSR) | 物体匹配（可选） |
| [PyTorch](https://pytorch.org)、[OpenUSD](https://openusd.org)、[Alembic](https://github.com/alembic/alembic)、Autodesk FBX SDK、[OpenColorIO](https://opencolorio.org)、[OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | 计算、色彩与文件格式 |

入镜自己的部分：把模型结果放进跟踪摄影机空间的整段解算（`shotspace/solver/`）、网页界面与任务队列、影视格式的导入导出。

## 许可

本项目以 [MIT](LICENSE) 许可开源。安装时下载的模型遵循各自许可（SAM License、DINOv3 License 等），详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
