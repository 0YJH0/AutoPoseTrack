# syntax=docker/dockerfile:1.7
# The upstream image contains the CUDA/PyTorch toolchain. The pinned source is
# supplied as third_party/FoundationPose and remains governed by its own license.
ARG FOUNDATIONPOSE_BASE=wenbowen123/foundationpose:latest
FROM ${FOUNDATIONPOSE_BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY third_party/FoundationPose /workspace/FoundationPose
WORKDIR /workspace/FoundationPose
RUN test -f run_demo.py \
    || (echo "third_party/FoundationPose is missing; add the pinned upstream checkout" >&2; exit 2)
RUN bash build_all.sh

COPY pyproject.toml README.md LICENSE /workspace/AutoPoseTrack/
COPY autoposetrack /workspace/AutoPoseTrack/autoposetrack
COPY configs /workspace/AutoPoseTrack/configs
COPY scripts /workspace/AutoPoseTrack/scripts
WORKDIR /workspace/AutoPoseTrack
RUN python -m pip install --no-deps -e .

CMD ["bash"]

