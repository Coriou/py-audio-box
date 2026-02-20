FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/Coriou/py-audio-box"
LABEL org.opencontainers.image.description="ML/audio toolbox — voice cloning, synthesis, separation"
LABEL org.opencontainers.image.licenses="MIT"

ARG POETRY_VERSION=1.8.3
ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000

# COMPUTE selects the PyTorch variant installed during the build.
# Values:
#   cpu    (default) — pure-CPU, runs on any host, zero extra requirements
#   cu124            — CUDA 12.4 wheels, enables GPU on Volta SM 7.0+ hosts
#
# IMPORTANT: declare this ARG *after* the expensive cached layers (apt, Poetry)
# so that changing it only invalidates the thin CUDA-swap layer and below —
# not the full image rebuild.  Override via docker-compose.gpu.yml.
# The ARG is re-declared just before its first use; this section is only a
# comment anchor to document the variable's purpose near the top of the file.

ENV \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_NO_INTERACTION=1 \
  # Install into system Python (no venv) — must run poetry install as root so
  # it has permission to write to /usr/local/lib/python3.11/site-packages
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR=/root/.cache/pypoetry

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  ffmpeg \
  sox \
  coreutils \
  tini \
  jq \
  nodejs \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry as root (pinned)
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION} \
  && /root/.local/bin/poetry --version

ENV PATH="/root/.local/bin:${PATH}"

# Copy only dependency manifests first (for layer caching)
COPY pyproject.toml poetry.lock* /app/
WORKDIR /app

# Install Python deps into system site-packages as root.
# BuildKit cache keeps the Poetry/pip download caches across rebuilds.
RUN --mount=type=cache,target=/root/.cache/pypoetry \
  --mount=type=cache,target=/root/.cache/pip \
  poetry install --no-root --sync

# ── GPU variant: upgrade torch/torchaudio to CUDA-enabled wheels ───────────────
# ARG declared here (not at the top) so that the apt + Poetry layers above are
# NOT invalidated when COMPUTE changes — only this layer and below re-run.
#
# For COMPUTE=cpu  this is a no-op (zero cost, zero extra download).
# For COMPUTE=cu121 the CPU wheels that Poetry just installed are replaced with
# CUDA-enabled equivalents.  All other packages remain identical across variants.
#
# We install the latest torch/torchaudio available on the requested CUDA index
# rather than trying to match the exact CPU-installed version.  The CPU wheels
# are published more frequently than the CUDA wheels, so an exact-version match
# would break whenever the CPU index races ahead (e.g. torch 2.10.0 on CPU but
# only 2.6.0+cu124 on the CUDA index).  Any version satisfying ^2.3.0 is fine.
ARG COMPUTE=cpu
RUN --mount=type=cache,target=/root/.cache/pip \
  if [ "${COMPUTE}" != "cpu" ]; then \
  echo "==> Upgrading torch/torchaudio → CUDA ${COMPUTE} wheels …" \
  && pip install --force-reinstall \
  "torch" "torchaudio" \
  --index-url "https://download.pytorch.org/whl/${COMPUTE}"; \
  fi

# Create non-root user and hand /app over to them
RUN groupadd -g ${APP_GID} ${APP_USER} \
  && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}

# Copy the rest of the source
COPY --chown=${APP_USER}:${APP_USER} . /app

USER ${APP_USER}

# Point all cache-aware tools (Torch Hub, HuggingFace, etc.) at the shared /cache mount
ENV XDG_CACHE_HOME=/cache

# Expose the compute variant so apps and diagnostic tooling can inspect it:
#   docker run --rm voice-tools:cuda env | grep TORCH_COMPUTE
# Note: value is baked in at build time; overriding COMPUTE at run-time has no effect.
ENV TORCH_COMPUTE=${COMPUTE}

# Default entrypoint: tini for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
