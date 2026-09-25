# Docker development and reproducibility

## Isolation policy

Docker is the primary environment boundary. The project does not install Conda
inside its own images. The core image uses Python 3.11 and pip metadata from
`pyproject.toml`; FoundationPose is isolated in a separate GPU image because its
CUDA/native dependency stack is substantially different.

Do not copy datasets, checkpoints, or experiment outputs into an image. Mount
them at runtime so image rebuilds are cheap and research artifacts persist.

## Current host status (2026-09-25)

- WSL2 is active.
- CUDA toolkit 11.8 and `nvcc` exist.
- GPU compute is **not available**: `/dev/dxg` is absent and `nvidia-smi` reports
  `GPU access blocked by the operating system`.
- Docker is **not available in this WSL distribution**; Docker Desktop reports
  that WSL integration must be enabled.

The CUDA toolkit does not prove that a GPU is accessible. Treat the preflight
script as the gate:

```bash
python -m scripts.check_environment --require-gpu --require-docker
```

## Fix WSL and Docker Desktop first

From an elevated Windows PowerShell:

```powershell
wsl --update
wsl --shutdown
```

Update the Windows NVIDIA driver, restart Windows, start Docker Desktop, select
the WSL 2 engine, and enable this distribution under
`Settings > Resources > WSL Integration`. Do not install a Linux NVIDIA display
driver inside WSL; WSL receives the driver interface from Windows.

After reopening WSL:

```bash
ls -l /dev/dxg
/usr/lib/wsl/lib/nvidia-smi
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
python -m scripts.check_environment --require-gpu --require-docker
```

## Core image

The core image is CPU-capable and independent of FoundationPose:

```bash
docker compose build core
docker compose run --rm core
docker compose run --rm core \
  python -m scripts.validate_config \
  configs/experiments/foundationpose_ycbv_smoke.yaml
```

## Live source development

The `dev` service bind-mounts the repository. Source edits on the host are
visible immediately and do not require an image rebuild:

```bash
docker compose --profile dev run --rm dev
python -m pytest
```

Rebuild after changing `pyproject.toml`, the Python version, or system packages:

```bash
docker compose build core
```

For a paper run, do not use the mutable development mount. Build a versioned
image from a clean commit and record its immutable digest:

```bash
AUTPOSETRACK_VERSION=$(git rev-parse --short HEAD) docker compose build core
docker image inspect autoposetrack-core:$(git rev-parse --short HEAD) \
  --format '{{index .RepoDigests 0}} {{.Id}}'
```

## FoundationPose image

Only proceed after GPU preflight passes. Pin FoundationPose as a submodule:

```bash
git submodule add https://github.com/NVlabs/FoundationPose.git third_party/FoundationPose
git -C third_party/FoundationPose checkout <REVIEWED_COMMIT>
git add .gitmodules third_party/FoundationPose
```

Set external paths and build:

```bash
export YCBV_ROOT=/absolute/path/to/ycbv
export FOUNDATIONPOSE_WEIGHTS=/absolute/path/to/foundationpose/weights
export AUTPOSETRACK_VERSION=$(git rev-parse --short HEAD)
docker compose --profile gpu build foundationpose
docker compose --profile gpu run --rm foundationpose nvidia-smi
```

The default FoundationPose base tag follows upstream documentation but is not
immutable. Before a reproducibility run, resolve it to a digest and set:

```bash
export FOUNDATIONPOSE_BASE='wenbowen123/foundationpose@sha256:<digest>'
```

FoundationPose source, weights, base image, renderers, and datasets retain their
own licenses. Review them before publishing an artifact.

