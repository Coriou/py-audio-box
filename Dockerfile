FROM python:3.11-slim-bookworm

ARG POETRY_VERSION=1.8.3
ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000

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

# Create non-root user and hand /app over to them
RUN groupadd -g ${APP_GID} ${APP_USER} \
  && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}

# Copy the rest of the source
COPY --chown=${APP_USER}:${APP_USER} . /app

USER ${APP_USER}

# Point all cache-aware tools (Torch Hub, HuggingFace, etc.) at the shared /cache mount
ENV XDG_CACHE_HOME=/cache

# Default entrypoint: tini for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
