FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/Coriou/py-audio-box"
LABEL org.opencontainers.image.description="ML/audio toolbox — voice cloning, synthesis, separation"
LABEL org.opencontainers.image.licenses="MIT"

ARG POETRY_VERSION=1.8.3

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
  rsync \
  ffmpeg \
  sox \
  coreutils \
  tini \
  jq \
  nodejs \
  vim \
  htop \
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

# Install vastai CLI in an isolated venv so its pinned transitive deps
# (notably python-dateutil) cannot downgrade app/runtime dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
  python -m venv /opt/vastai \
  && /opt/vastai/bin/pip install --upgrade pip \
  && /opt/vastai/bin/pip install vastai \
  && ln -sf /opt/vastai/bin/vastai /usr/local/bin/vastai

# ── GPU variant: upgrade torch/torchaudio to CUDA-enabled wheels ───────────────
# ARG declared here (not at the top) so that the apt + Poetry layers above are
# NOT invalidated when COMPUTE changes — only this layer and below re-run.
#
# For COMPUTE=cpu  this is a no-op (zero cost, zero extra download).
# For COMPUTE=cu121 the CPU wheels that Poetry just installed are replaced with
# CUDA-enabled equivalents.  All other packages remain identical across variants.
#
# torch / flash-attn install strategy per COMPUTE variant:
#
#   cpu    — no-op; CPU wheels already installed by Poetry above.
#
#   cu124  — pin torch==2.6.0 then build flash-attn from source with
#            --no-build-isolation.  Building from source (rather than using
#            a pre-built wheel) guarantees the C++ ABI of flash_attn's CUDA
#            extension matches the torch it is compiled against.  Pre-built
#            wheels with cxx11abiFALSE/TRUE suffixes are unreliable: a
# torch / flash-attn install strategy per COMPUTE variant:
#
#   cpu    — no-op; CPU wheels already installed by Poetry above.
#
#   cu124  — pin torch==2.6.0, then install flash-attn from the pre-built wheel.
#            The wheel variant to try is cxx11abiTRUE.  Counterintuitively,
#            pip-distributed torch 2.6.0+cu124 on Linux reports
#            _GLIBCXX_USE_CXX11_ABI=False via Python, but its shared libraries
#            (libc10.so, etc.) export the cxx11 ABI variants of c10::Error and
#            similar symbols.  The "cxx11abiFALSE" wheel references non-cxx11
#            symbol names which are absent, causing "undefined symbol" at import.
#            Using cxx11abiTRUE resolves this.
#            If the pre-built wheel still fails on a new instance, fall back to
#            building from source (see comments in vast-deploy.sh).
#
#   cu128+ — install latest torch from the given index.  No flash-attn wheel
#            exists for these variants (SM 12.0 / Blackwell consumer GPUs have
#            separate kernel-dispatch issues regardless).
ARG COMPUTE=cpu
ARG TORCH_CU124_VERSION=2.6.0
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
  if [ "${COMPUTE}" = "cu124" ]; then \
  echo "==> Upgrading torch/torchaudio → CUDA cu124 (pinned ${TORCH_CU124_VERSION}) …" \
  && pip install --force-reinstall \
  "torch==${TORCH_CU124_VERSION}" "torchaudio==${TORCH_CU124_VERSION}" \
  --index-url "https://download.pytorch.org/whl/cu124" \
  && echo "==> Installing flash-attn (cxx11abiTRUE wheel for cu12+torch${TORCH_CU124_VERSION}+cp311) …" \
  && pip install --no-build-isolation "${FLASH_ATTN_WHEEL}"; \
  elif [ "${COMPUTE}" != "cpu" ]; then \
  echo "==> Upgrading torch/torchaudio → CUDA ${COMPUTE} wheels (latest) …" \
  && pip install --force-reinstall \
  "torch" "torchaudio" \
  --index-url "https://download.pytorch.org/whl/${COMPUTE}"; \
  fi

# Copy the rest of the source
COPY . /app

# Create persistent-mount targets so they exist even if no volume is attached.
# vast.ai persistent storage should be mounted at /cache to skip model re-downloads.
# Disable vast.ai's auto-tmux wrapper (it intercepts SSH sessions and breaks
# non-interactive commands).  The file must exist in root's home at image build
# time so it is present before the first SSH login.
RUN mkdir -p /work /cache && touch /root/.no_auto_tmux

# Runtime environment:
#   XDG_CACHE_HOME / HF_HOME / TORCH_HOME all point to /cache so model weights
#   land in the persistent volume when one is attached.
#   TORCH_DEVICE defaults to cuda; override with -e TORCH_DEVICE=cpu for CPU-only.
#   HF_HUB_DISABLE_XET_TRANSPORT avoids a slow experimental transfer path.
ENV XDG_CACHE_HOME=/cache \
  HF_HOME=/cache/huggingface \
  TORCH_HOME=/cache/torch \
  TORCH_DEVICE=cuda \
  HF_HUB_DISABLE_XET_TRANSPORT=1

# Expose the compute variant so apps and diagnostic tooling can inspect it:
#   docker run --rm voice-tools:cuda env | grep TORCH_COMPUTE
# Note: value is baked in at build time; overriding COMPUTE at run-time has no effect.
ENV TORCH_COMPUTE=${COMPUTE}

WORKDIR /app

# Default entrypoint: tini for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
