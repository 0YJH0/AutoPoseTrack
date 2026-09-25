# Environment audit

Captured: 2026-09-25 (Asia/Shanghai)

## Observed host

| Item | Observed value |
|---|---|
| Kernel | Linux 5.15.167.4-microsoft-standard-WSL2, x86_64 |
| CPU | Intel Core i9-14900HX, 32 logical CPUs / 16 cores |
| Memory | 15 GiB total, 4 GiB swap |
| Workspace filesystem | 1007 GiB total, 353 GiB available |
| Python | 3.9.12 at `/home/yjh/anaconda3/bin/python3` |
| Conda | 4.12.0 |
| pip | 21.2.4 |
| GCC | 11.4.0 |
| Git | 2.34.1 |
| GPU | **Unknown/unavailable**: NVML reports GPU access blocked by OS |
| WSL GPU device | `/dev/dxg` is absent; CUDA compute is unavailable |
| CUDA toolkit | `nvcc` 11.8 is installed; this does not imply GPU access |
| Docker | CLI unavailable; Docker Desktop WSL integration is not enabled |
| Repository state | Empty scaffold at audit start; `.git` is not a valid Git repository |

## Re-run commands

```bash
uname -a
lscpu
free -h
df -h .
python3 --version
conda --version
pip --version
gcc --version
git --version
nvidia-smi
git status --short --branch
```

## Consequences

- No CUDA, VRAM, driver, or baseline runtime claim can currently be verified.
- Docker and GPU containers cannot run until Docker Desktop integration and the
  WSL GPU device are restored; see `docs/docker.md`.
- The existing Python 3.9 base environment must not be reused as the research
  environment; FoundationPose upstream currently documents a distinct Python
  environment and compiled CUDA dependencies.
- The directory is not currently a functional Git worktree, so commit capture
  and submodules cannot be used until repository initialization is resolved.
- Sixteen GiB host RAM may constrain parallel data loading and template caches;
  measure peak host/GPU memory during the smoke test rather than guessing.
