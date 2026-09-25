# Docker development and reproducibility

## Isolation policy

Docker is the primary environment boundary. The project does not install Conda
inside its own images. The core image uses Python 3.11 and pip metadata from
`pyproject.toml`; MegaPose is isolated in a separate GPU image because its
CUDA/native dependency stack is substantially different.

Do not copy datasets, checkpoints, or experiment outputs into an image. Mount
them at runtime so image rebuilds are cheap and research artifacts persist.

## Verified host status (2026-09-25)

- WSL2 and Docker Desktop WSL integration are active.
- CUDA toolkit 11.8 and `nvcc` exist on the WSL host.
- RTX 4060 Laptop GPU (8188 MiB, driver 576.02) is visible from ordinary WSL.
- GPU passthrough was verified with the CUDA 12.4.1 Ubuntu 22.04 container.
- Docker Desktop 4.43.1 / Engine 28.3.0 are available.
- Core and training images build successfully and pass their checks.

The CUDA toolkit alone does not prove that a GPU is accessible. Treat the
preflight script and a CUDA-container smoke test as gates. A restricted command
sandbox may intentionally hide `/dev/dxg`; run hardware preflight in an ordinary
WSL terminal:

```bash
python -m scripts.check_environment --require-gpu --require-docker
```

## WSL and Docker Desktop recovery procedure

From an elevated Windows PowerShell:

```powershell
wsl --update
wsl --shutdown
```

Update the Windows NVIDIA driver, restart Windows, start Docker Desktop, select
the WSL 2 engine, and enable this distribution under
`Settings > Resources > WSL Integration`. Do not install a Linux NVIDIA display
driver inside WSL; WSL receives the driver interface from Windows.

If GPU or Docker access regresses, reopen WSL and run:

```bash
ls -l /dev/dxg
/usr/lib/wsl/lib/nvidia-smi
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
python -m scripts.check_environment --require-gpu --require-docker
```

## Core image

The core image is CPU-capable and independent of MegaPose:

```bash
docker compose build core
docker compose run --rm core
docker compose run --rm core \
  python -m scripts.validate_config \
  configs/experiments/megapose_rgb_ycbv_smoke.yaml
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

## Training image

Training and plotting dependencies are isolated from the minimal core image:

```bash
export FEATURE_DATA_ROOT=/absolute/path/to/generated/features
docker compose build training
docker compose run --rm training
```

The feature root is mounted read-only at `/data/features`; checkpoints and
metrics are written under the mounted `outputs/` directory. Add future PyTorch
dependencies to the `train` optional dependency or a dedicated GPU training
target, not to the core runtime.

For a paper run, do not use the mutable development mount. Build a versioned
image from a clean commit and record its immutable digest:

```bash
AUTPOSETRACK_VERSION=$(git rev-parse --short HEAD) docker compose build core
docker image inspect autoposetrack-core:$(git rev-parse --short HEAD) \
  --format '{{index .RepoDigests 0}} {{.Id}}'
```

## MegaPose RGB image

Pin MegaPose and its submodules before building:

```bash
git submodule add https://github.com/megapose6d/megapose6d.git third_party/MegaPose
git -C third_party/MegaPose submodule update --init
git -C third_party/MegaPose checkout <REVIEWED_COMMIT>
git add .gitmodules third_party/MegaPose
```

Set external paths and build:

```bash
export YCBV_ROOT=/absolute/path/to/ycbv
export MEGAPOSE_DATA_ROOT=/absolute/path/to/megapose-data
export AUTPOSETRACK_VERSION=$(git rev-parse --short HEAD)
docker compose --profile gpu build megapose
docker compose --profile gpu run --rm megapose nvidia-smi
```

The default MegaPose base tag follows upstream documentation but is not
immutable. Before a reproducibility run, resolve it to a digest and set:

```bash
export MEGAPOSE_BASE='ylabbe/megapose6d@sha256:<digest>'
```

MegaPose source, weights, base image, renderers, and datasets retain their
own licenses. Review them before publishing an artifact.
