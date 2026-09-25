# syntax=docker/dockerfile:1.7
# The official upstream image provides MegaPose's CUDA/PyTorch/rendering stack.
# Source is supplied as a pinned submodule and retains its Apache-2.0 license.
ARG MEGAPOSE_BASE=ylabbe/megapose6d:latest
FROM ${MEGAPOSE_BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEGAPOSE_DATA_DIR=/megapose-data

COPY third_party/MegaPose /workspace/MegaPose
WORKDIR /workspace/MegaPose
RUN test -f setup.py -o -f pyproject.toml \
    || (echo "third_party/MegaPose is missing; add a pinned upstream checkout" >&2; exit 2)
RUN python -m pip install --no-deps -e .

COPY pyproject.toml README.md LICENSE /workspace/AutoPoseTrack/
COPY autoposetrack /workspace/AutoPoseTrack/autoposetrack
COPY configs /workspace/AutoPoseTrack/configs
COPY scripts /workspace/AutoPoseTrack/scripts
WORKDIR /workspace/AutoPoseTrack
RUN python -m pip install --no-deps -e .

CMD ["bash"]
