# ShotSpace · 入镜

**[中文说明](#中文说明)**

ShotSpace puts the people in a live-action shot into 3D, in the space of the shot's tracked camera.
You give it an undistorted image sequence, the camera your matchmove already solved, and (optionally) a scan of
the set; it finds and tracks every person, estimates each body with SAM 3D Body under the camera's real lens, and
solves every person over the whole shot so they stand where they really were. The results export as USD (skeleton
and point cache), FBX and Alembic, ready for Maya, Houdini, Nuke or any USD tool.

It runs as a web service on one GPU computer; artists use it from Chrome on their own computers. Each computer
sees and works on its own tasks; an admin page shows everything.

## What you need

- An NVIDIA GPU with about 24 GB of memory (RTX 4090 class), driver 570 or newer (CUDA 12.8).
- Linux x86_64 (glibc 2.28 or newer) or Windows 10 / 11 x64.
- About 30 GB of free disk (about 45 GB more with object matching), and an internet connection for the setup.
- Chrome on the computers that use it.

## Install

```bash
git clone https://github.com/vfxli/ShotSpace.git
cd ShotSpace
./setup.sh            # Linux
setup.bat             # Windows (double-click, or run in a terminal)
```

The setup wizard asks a few questions (language, download sources, which parts, where the models come from) and
then installs everything **inside the ShotSpace folder**: its own copy of uv and Python, the environments, the
source code of the models and their weights. Nothing is installed system-wide; PATH and the registry are not
changed. To uninstall, delete the folder.

- **Download sources.** Every download has several sources and the wizard falls back to the next one when one fails
  or is slow: PyPI (pypi.org, Aliyun, Tsinghua, USTC), PyTorch (download.pytorch.org, SJTU, NJU), Python
  (GitHub, npmmirror), source code (GitHub, gh-proxy, ghfast), model weights (ModelScope, HuggingFace,
  hf-mirror). Choose "measure" and it picks the fastest. Every file is checked against the SHA-256 in
  `install/sources.json`, so a mirror can only be slow or missing, never wrong.
- **Interrupted?** Run the setup again; it continues where it stopped (downloads resume).
- **Model weights.** SAM 3D Body and SAM 3.1 are gated on HuggingFace: request access on
  [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and
  [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), then give the wizard a read token (it is used for the
  download only and never stored; or set `HF_TOKEN`). ModelScope carries the same files (checked byte for byte) and
  needs no account. If you already have the weights, point the wizard at the folder.
- **Non-interactive:** `./setup.sh --yes [--region cn|global] [--weights modelscope|huggingface|<folder>] [--objects] [--no-export]`.
- **Object matching** (optional, Linux only) adds SAM 3D Objects, MoGe and TripoSR (about 15 GB of models).
- **ABC / FBX export** uses prebuilt modules downloaded from this repository's releases (no compiler needed).
  USD export needs nothing extra.

## Run

```bash
./shotspace.sh start          # Linux; first start asks for the port and an admin password
shotspace.bat start           # Windows
```

Open `http://<server>:8765/` in Chrome (the admin page is `/admin`). `./shotspace.sh stop`, `status` and
`doctor` do what they say.

## Using it

1. Create a task, load the image sequence (EXR in ACEScg by default, colour handled with OCIO; PNG/JPG/TIFF work too)
   and grade it if needed (the grade is used for solving as well).
2. Load the tracked camera (`.abc`, `.fbx` or `.usd`); its per-frame lens and position are used exactly as they are.
3. Solve people: ShotSpace finds and tracks everyone, you pick who to solve. Solving adds up: solve one person, look,
   then add more.
4. Optionally load a scan of the set (`.abc`, `.fbx`, `.usd`); with a scan people stand on the real ground and get
   their true size. Shots without visible ground work without one.
5. Export the people you want as USD skeleton, USD point cache, FBX or Alembic (metres).

## How it works

1. **Find and follow people** - SAM 3.1 tracks every person through the shot (forward and backward), one outline per
   person per frame.
2. **Per-frame body** - SAM 3D Body estimates pose and body shape on every frame under the camera's real
   intrinsics, each person with their own outline.
3. **Whole-shot solve** - per-frame results are only a start: each person is solved over the whole shot with one
   body shape, their position constrained by the exact camera on every frame and, with a scan, by the ground.

Code: `shotspace/solver/workers/people_tracker.py`, `shotspace/solver/workers/sam3d_worker.py`,
`shotspace/solver/trajectory.py`, `shotspace/solver/workers/body_solve.py`.

## Tests

```bash
.venv/bin/python tests/test_shotspace.py          # Linux
.venv\Scripts\python.exe tests\test_shotspace.py  # Windows
```

## Licence

ShotSpace is released under the [MIT licence](LICENSE). The models it downloads keep their own licences
(SAM License, DINOv3 License, and for object matching also MIT); the wizard lists them and asks you to accept them
before installing. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 中文说明

入镜（ShotSpace）把实拍镜头里的人解算成三维，放进镜头已跟踪好的摄影机所在的空间。输入去畸变的序列图、
已经跟踪好的摄影机、（可选）场景扫描；它找到并跟踪画面里的每个人，用 SAM 3D Body 在摄影机的真实镜头参数下
估计每个人的身体，再对每个人整段镜头一起解算，让人站在真实的位置上。结果导出为 USD（骨架、点缓存）、FBX、
Alembic，可直接用于 Maya、Houdini、Nuke 等软件。

它作为网页服务运行在一台带显卡的电脑上，大家在自己电脑的 Chrome 里使用。每台电脑只看到、只能操作自己的任务，
管理后台能看到全部任务。

### 需要什么

- NVIDIA 显卡，显存约 24 GB（RTX 4090 级别），驱动 570 以上（CUDA 12.8）。
- Linux x86_64（glibc 2.28 以上）或 Windows 10 / 11 64 位。
- 约 30 GB 剩余空间（装物体匹配再加约 45 GB），安装时需要联网。
- 使用的电脑上装有 Chrome。

### 安装

```bash
git clone https://github.com/vfxli/ShotSpace.git
cd ShotSpace
./setup.sh            # Linux
setup.bat             # Windows（双击或在终端运行）
```

安装向导会问几个问题（语言、下载源、装哪些部分、模型从哪里下），然后把所有东西都装进 **ShotSpace 文件夹内**：
自带的 uv 和 Python、各个运行环境、模型源码和权重。不装任何系统级的东西，不改 PATH，不写注册表。
卸载就是删掉这个文件夹。

- **下载源**：每样东西都有多个下载源，一个失败或太慢就自动换下一个：PyPI（官方、阿里云、清华、中科大）、
  PyTorch（官方、上海交大、南京大学）、Python（GitHub、npmmirror）、源码（GitHub、gh-proxy、ghfast）、
  模型权重（魔搭 ModelScope、HuggingFace、hf-mirror）。选"自动测速"会挑最快的。每个文件都按
  `install/sources.json` 里记录的 SHA-256 校验，镜像只会慢或缺，不会给错文件。
- **中途断了**：重新运行安装程序，会从断点继续（下载也会续传）。
- **模型权重**：SAM 3D Body 和 SAM 3.1 在 HuggingFace 上需要先申请：在
  [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) 和
  [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) 页面申请并同意许可，再把只读 token 给安装向导
  （只用于下载，不会保存；也可以设置环境变量 `HF_TOKEN`）。魔搭 ModelScope 上有完全相同的文件（逐字节核对过），
  不需要账号，国内推荐用它。已经有权重文件的，选"本机已有的文件夹"。
- **不问问题直接装**：`./setup.sh --yes [--region cn|global] [--weights modelscope|huggingface|文件夹] [--objects] [--no-export]`。
- **物体匹配**（可选，只支持 Linux）：多装 SAM 3D Objects、MoGe、TripoSR，模型约 15 GB。
- **ABC / FBX 导出**：用本仓库 Release 里编译好的模块，不需要装编译器。USD 导出不需要额外安装。

### 启动

```bash
./shotspace.sh start          # Linux；第一次启动会让你设置端口和后台密码
shotspace.bat start           # Windows
```

在 Chrome 里打开 `http://<服务器地址>:8765/`，管理后台是 `/admin`。另有 `stop`、`status`、`doctor` 命令。

### 使用步骤

1. 新建任务，导入序列图（默认 ACEScg 的 EXR，经 OCIO 转换；也支持 PNG/JPG/TIFF），需要时调色（调色结果同时用于解算）。
2. 导入摄影机（`.abc` / `.fbx` / `.usd`），逐帧内外参原样使用，不做反求。
3. 人物解算：自动找出并跟踪所有人，你选择要解算谁。解算是累加的，可以先解一个人，看完再加别人。
4. 可选：导入场景扫描（`.abc` / `.fbx` / `.usd`），有扫描时人站在真实地面上、得到真实大小；看不到地面的镜头不导入也能正常解算。
5. 选择要导出的人，导出 USD 骨架、USD 点缓存、FBX 或 Alembic（单位米）。

### 许可

入镜以 [MIT 许可](LICENSE) 开源。安装时下载的模型各有自己的许可（SAM License、DINOv3 License，物体匹配还有 MIT），
安装向导会列出并请你确认。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
