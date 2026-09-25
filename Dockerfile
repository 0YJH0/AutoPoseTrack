# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ARG AUTPOSETRACK_VERSION=dev
LABEL org.opencontainers.image.title="AutoPoseTrack Core" \
      org.opencontainers.image.description="Dependency-light research infrastructure" \
      org.opencontainers.image.source="https://github.com/0YJH0/AutoPoseTrack" \
      org.opencontainers.image.version="${AUTPOSETRACK_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/AutoPoseTrack

RUN apt-get update \
    && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first so source edits reuse the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY autoposetrack ./autoposetrack
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[dev]"

COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "-q"]

FROM base AS training
RUN python -m pip install ".[train,viz]"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "-q"]
