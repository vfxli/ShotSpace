# Third-party software and models

ShotSpace's own code is MIT licensed (see `LICENSE`). This repository does not contain third-party model code or
weights: the setup wizard downloads them from their publishers (or from mirrors of the same files, checked by
SHA-256) into `third_party/`, and asks you to accept their licences first.

## Downloaded by the setup wizard

| Component | Used for | Licence |
|---|---|---|
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) code and the `facebook/sam-3d-body-dinov3` weights (incl. the MHR body model) | per-frame body estimation | SAM License |
| [DINOv3](https://github.com/facebookresearch/dinov3) code | backbone of SAM 3D Body | DINOv3 License |
| [SAM 3](https://github.com/facebookresearch/sam3) code (with ShotSpace's patch `install/patches/sam3-shotspace.patch`) and the `facebook/sam3.1` weights | finding and tracking people | SAM License |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) code and weights (optional) | object matching | SAM License |
| [MoGe](https://github.com/microsoft/MoGe) code and `Ruicheng/moge-vitl` weights (optional) | object matching | MIT |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) code and weights (optional) | object matching | MIT |
| PyTorch and the Python packages listed in `install/requirements/` | runtime | their own licences (BSD, MIT, Apache 2.0, ...) |

## Prebuilt export modules (GitHub release `native-*`)

The ABC / FBX export modules are prebuilt from fixed upstream commits and downloaded by the setup wizard from
this repository's release `native-1` (checked by SHA-256). Each zip contains a `BUNDLE_INFO.txt` with exact
versions, build steps and the full licence texts.

| Component | Licence |
|---|---|
| Imath / PyImath 3.2.3 | BSD-3-Clause |
| Alembic / PyAlembic 1.8.12 | BSD-3-Clause |
| Boost.Python 1.92.0 | Boost Software License 1.0 |
| pybind11 3.1.0 | BSD-3-Clause |
| Microsoft Visual C++ runtime DLLs (Windows zip) | Microsoft Visual C++ redistributable terms |
| `fbxio` (ShotSpace's FBX module), statically linked with the Autodesk FBX SDK 2020.3.10 | ShotSpace's code: MIT; the FBX SDK code in it: Autodesk FBX SDK 2020 License and Services Agreement, provided "as is" |

The MIT licence of ShotSpace does not apply to the Autodesk FBX SDK code inside `fbxio`. Notice required by the
Autodesk FBX SDK licence:

> This software contains Autodesk® FBX® code developed by Autodesk, Inc. Copyright 2019 Autodesk, Inc. All
> rights, reserved. Such code is provided "as is" and Autodesk, Inc. disclaims any and all warranties, whether
> express or implied, including without limitation the implied warranties of merchantability, fitness for a
> particular purpose or non-infringement of third party rights. In no event shall Autodesk, Inc. be liable for
> any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to,
> procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however
> caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or
> otherwise) arising in any way out of such code.
