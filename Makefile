# Toolbox Makefile — shortcuts for common tasks
# Usage: make <target> [ARGS="...extra flags..."]
#
# Most workflows go through ./run <app> instead, but this is handy for
# build / shell / one-liner invocations.
#
# GPU targets require NVIDIA Container Toolkit and the CUDA image built first.
# See docker-compose.gpu.yml for full GPU setup instructions.

.DEFAULT_GOAL := help

# ── infrastructure ─────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build (or rebuild) the CPU toolbox image  [default]
	docker compose build

.PHONY: build-gpu
build-gpu:  ## Build the GPU (CUDA 12.4) image variant  →  voice-tools:cuda
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build

.PHONY: publish
publish:  ## Build + push cpu (latest) and cuda images to GHCR
	./scripts/publish.sh

.PHONY: publish-all
publish-all:  ## Build + push all three variants: latest, cuda, cuda128
	./scripts/publish.sh all

.PHONY: publish-cpu
publish-cpu:  ## Build + push CPU image to GHCR  →  voice-tools:latest
	./scripts/publish.sh cpu

.PHONY: publish-cuda
publish-cuda:  ## Build + push CUDA 12.4 image to GHCR  →  voice-tools:cuda
	./scripts/publish.sh cuda

.PHONY: publish-cuda128
publish-cuda128:  ## Build + push CUDA 12.8 image to GHCR  →  voice-tools:cuda128  (Blackwell)
	./scripts/publish.sh cuda128

.PHONY: build-no-cache
build-no-cache:  ## Force a clean CPU rebuild (no layer cache)
	docker compose build --no-cache

.PHONY: build-gpu-no-cache
build-gpu-no-cache:  ## Force a clean GPU rebuild (no layer cache)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache

.PHONY: shell
shell:  ## Interactive bash shell inside the CPU toolbox
	docker compose run --rm toolbox bash

.PHONY: shell-gpu
shell-gpu:  ## Interactive bash shell inside the GPU toolbox
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm toolbox bash

.PHONY: clean-work
clean-work:  ## Delete all output files in ./work/ (keeps cache)
	rm -rf work/clip_*.wav

.PHONY: clean-cache
clean-cache:  ## Wipe the shared cache (forces model re-download on next run)
	rm -rf cache/

# ── remote (vast.ai) ──────────────────────────────────────────────────────────
# Remote sync targets are intentionally opt-in:
#   make pull REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22222
#   make push-code REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22222

REMOTE_HOST ?=
REMOTE_PORT ?= 22

define _require_remote_host
	@if [ -z "$(REMOTE_HOST)" ]; then \
		echo "ERROR: REMOTE_HOST is not set."; \
		echo "  Example: make pull REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=22222"; \
		exit 1; \
	fi
endef

.PHONY: pull
pull:  ## Pull /work/ from remote instance → work_remote/  [REMOTE_HOST=... REMOTE_PORT=...]
	$(_require_remote_host)
	mkdir -p work_remote
	rsync -avz --progress -e "ssh -p $(REMOTE_PORT)" $(REMOTE_HOST):/work/ work_remote/

.PHONY: push-code
push-code:  ## Push local code changes to remote /app/  [REMOTE_HOST=... REMOTE_PORT=...]
	$(_require_remote_host)
	rsync -avz --progress \
	  --exclude='.git' --exclude='work/' --exclude='work_remote/' --exclude='cache/' \
	  -e "ssh -p $(REMOTE_PORT)" ./ $(REMOTE_HOST):/app/

# ── vast.ai automation ─────────────────────────────────────────────────────────
# Provision a cloud GPU, run tasks, pull results, then auto-destroy.
#
# Quick start:
#   export VAST_API_KEY=xxxx          # one-time — get from cloud.vast.ai/console/cli
#   make vast-run TASKS=my-jobs.txt   # run tasks from a file
#   make vast-run ARGS='-- voice-synth speak --voice myvoice --text "Hello"'
#   make vast-shell                   # provision + interactive SSH shell
#   make vast-status                  # list your running instances
#   make vast-destroy ID=12345        # destroy a specific instance
#   make vast-search                  # show best-value qualifying offers
#
# Env overrides (all optional):
#   VAST_QUERY   custom GPU search query (default: reliable Volta+ ≥20 GB VRAM)
#   VAST_IMAGE   Docker image to deploy  (default: ghcr.io/coriou/voice-tools:cuda)
#   VAST_DISK    disk size in GB         (default: 60)
#   VAST_REPO    Git repo to clone       (default: https://github.com/Coriou/py-audio-box)
#
# Pass extra flags to vast-deploy.sh via ARGS:
#   make vast-run ARGS='--no-clone --keep' TASKS=my-jobs.txt

VAST_CLI    := ./scripts/vast
DEPLOY_SCRIPT := ./scripts/vast-deploy.sh
TASKS       ?=
ID          ?=

.PHONY: vast-search
vast-search:  ## Show best-value qualifying GPU offers on vast.ai (sorted by DLPerf/$)
	$(VAST_CLI) search offers \
	  'reliability > 0.98 gpu_ram >= 20 compute_cap >= 700 inet_down >= 200 disk_space >= 50 rented=False' \
	  --order 'dlperf_per_dphtotal-'

.PHONY: vast-status
vast-status:  ## Show your currently running vast.ai instances
	$(VAST_CLI) show instances

.PHONY: vast-run
vast-run:  ## Provision GPU → run tasks → pull results → destroy  [TASKS=file | ARGS='-- cmd']
	@if [ -n "$(TASKS)" ]; then \
	  $(DEPLOY_SCRIPT) --tasks "$(TASKS)" $(ARGS); \
	else \
	  $(DEPLOY_SCRIPT) $(ARGS); \
	fi

.PHONY: vast-shell
vast-shell:  ## Provision a GPU instance and open an interactive SSH shell
	$(DEPLOY_SCRIPT) --shell $(ARGS)

.PHONY: vast-destroy
vast-destroy:  ## Destroy a specific instance by ID  [ID=12345]
	@if [ -z "$(ID)" ]; then \
	  echo "ERROR: ID is not set.  Example: make vast-destroy ID=12345"; exit 1; \
	fi
	$(VAST_CLI) destroy instance $(ID)

.PHONY: vast-pull
vast-pull:  ## Pull /work from a running instance  [ID=12345 JOB=name]
	@if [ -z "$(ID)" ]; then \
	  echo "ERROR: ID is not set.  Example: make vast-pull ID=12345 JOB=myjob"; exit 1; \
	fi
	$(eval _CONN := $(shell $(VAST_CLI) ssh-url $(ID) 2>/dev/null))
	$(eval _HOST := $(shell echo "$(_CONN)" | sed 's|ssh://[^@]*@||' | cut -d: -f1))
	$(eval _PORT := $(shell echo "$(_CONN)" | sed 's|.*:||'))
	mkdir -p work_remote/$(JOB)
	rsync -avz --progress \
	  -e "ssh -p $(_PORT) -o StrictHostKeyChecking=no" \
	  root@$(_HOST):/work/ work_remote/$(JOB)/

# ── apps ───────────────────────────────────────────────────────────────────────

.PHONY: voice-split
voice-split:  ## Run voice-split. Usage: make voice-split ARGS='--url "..." --clips 5'
	./run voice-split $(ARGS)

.PHONY: voice-clone
voice-clone:  ## Run voice-clone. Usage: make voice-clone ARGS='synth --ref-audio /work/x.wav --text "Hello"'
	./run voice-clone $(ARGS)

.PHONY: voice-synth
voice-synth:  ## Run voice-synth. Usage: make voice-synth ARGS='speak --voice <id> --text "Hello"'
	./run voice-synth $(ARGS)

.PHONY: voice-register
voice-register:  ## One-shot register. Usage: make voice-register ARGS='--url "..." --voice-name slug --text "Hello"'
	./run voice-register $(ARGS)

.PHONY: smoke-matrix
smoke-matrix:  ## Phase-5 smoke matrix (capabilities + clone + built-in + designed). Set SMOKE_SKIP_DESIGN=1 to skip design.
	./scripts/smoke-matrix.sh

# ── help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
