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

.PHONY: publish-cuda-base
publish-cuda-base:  ## Build + push CUDA base image (heavy layer: torch+flash-attn)  →  voice-tools:cuda-base
	./scripts/publish.sh cuda-base

.PHONY: publish-cuda
publish-cuda:  ## Build + push thin CUDA app image (FROM cuda-base + app src, ~50 MB)  →  voice-tools:cuda
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

.PHONY: vast-remote-test
vast-remote-test:  ## Provision GPU → run full remote test suite (tests/remote/) → rsync results → destroy
	# Runs tests/remote/run-all.sh on a fresh GPU instance.
	# Override test behaviour:
	#   ONLY="01 03"     run only specific suites (01-06)
	#   SKIP="05"        skip specific suites
	#   SKIP_SLOW=1      skip slow sections inside suites
	#   PARALLEL_N=4     parallel jobs in the stress test (default 6)
	#   TARGET=gpu       benchmark label (default: gpu)
	# Extra deploy flags via ARGS:   make vast-remote-test ARGS="--keep"
	$(DEPLOY_SCRIPT) \
	  --push-cache ./cache/voices \
	  $(ARGS) \
	  -- '!ONLY="$(ONLY)" SKIP="$(SKIP)" SKIP_SLOW="$(SKIP_SLOW)" PARALLEL_N="$(PARALLEL_N)" TARGET="$(TARGET)" RUN_ID="$(RUN_ID)" bash /app/tests/remote/run-all.sh'

# Bring up a synonym
.PHONY: synth-bench-gpu
synth-bench-gpu: vast-remote-test  ## Alias for vast-remote-test

ONLY        ?=
SKIP        ?=
SKIP_SLOW   ?= 0
PARALLEL_N  ?= 6
TARGET      ?= gpu
RUN_ID      ?= $(shell date +%Y%m%d_%H%M%S)

.PHONY: synth-bench-cpu
synth-bench-cpu:  ## Run the synthesis benchmark suite on the LOCAL CPU Docker container
	# Runs TARGET=cpu (float32) inside the local CPU toolbox Docker container.
	# Results saved to ./work/bench-tests/ on the host.
	# Override: make synth-bench-cpu ONLY="01 02" SKIP_SLOW=1
	docker compose run --rm \
	  -e TARGET=cpu \
	  -e DTYPE=float32 \
	  -e SKIP_SLOW="$(SKIP_SLOW)" \
	  -e ONLY="$(ONLY)" \
	  -e SKIP="$(SKIP)" \
	  -e PARALLEL_N="$(PARALLEL_N)" \
	  -e RUN_ID="$(RUN_ID)" \
	  toolbox bash /app/tests/remote/run-all.sh

# ROG GPU machine: Windows + WSL2 + Docker (CUDA 12.8 image)
ROG_HOST    ?= 100.81.65.12
ROG_PORT    ?= 1337
ROG_USER    ?= Benjamin
ROG_WSL     ?= ubuntu
ROG_REPO    ?= /mnt/c/Users/Benjamin/py-audio-box

.PHONY: synth-bench-rog
synth-bench-rog:  ## Run benchmark on the ROG GPU machine via SSH + WSL  [ROG_HOST=... ROG_PORT=...]
	# 1. Push latest voices to ROG
	@echo "==> Syncing voices to ROG ($(ROG_USER)@$(ROG_HOST):$(ROG_PORT))..."
	rsync -avz --progress \
	  -e "ssh -p $(ROG_PORT)" \
	  ./cache/voices/ $(ROG_USER)@$(ROG_HOST):$(ROG_REPO)/cache/voices/
	# 2. Run the benchmark inside the GPU docker on ROG
	@echo "==> Running benchmark on ROG..."
	ssh -p $(ROG_PORT) $(ROG_USER)@$(ROG_HOST) \
	  "wsl -d $(ROG_WSL) -u root bash -c \
	    'cd $(ROG_REPO) && \
	     docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
	       -e TARGET=rog -e DTYPE=bfloat16 \
	       -e SKIP_SLOW=\"$(SKIP_SLOW)\" -e ONLY=\"$(ONLY)\" -e SKIP=\"$(SKIP)\" \
	       -e PARALLEL_N=\"$(PARALLEL_N)\" -e RUN_ID=\"$(RUN_ID)\" \
	       toolbox bash /app/tests/remote/run-all.sh'"
	# 3. Pull results back
	@echo "==> Pulling results from ROG..."
	mkdir -p work_remote/rog-$(RUN_ID)
	rsync -avz --progress \
	  -e "ssh -p $(ROG_PORT)" \
	  $(ROG_USER)@$(ROG_HOST):$(ROG_REPO)/work/remote-tests/ \
	  work_remote/rog-$(RUN_ID)/

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

.PHONY: job-runner
job-runner:  ## Run job-runner. Usage: make job-runner ARGS='plan /app/jobs.example.yaml'
	./run job-runner $(ARGS)

.PHONY: jobs-plan
jobs-plan:  ## Validate and preview jobs YAML. Usage: make jobs-plan FILE=/app/jobs.example.yaml
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE is not set. Example: make jobs-plan FILE=/app/jobs.example.yaml"; \
		exit 1; \
	fi
	./run job-runner plan "$(FILE)" $(ARGS)

.PHONY: jobs-enqueue
jobs-enqueue:  ## Enqueue jobs YAML. Usage: make jobs-enqueue FILE=/app/jobs.example.yaml
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE is not set. Example: make jobs-enqueue FILE=/app/jobs.example.yaml"; \
		exit 1; \
	fi
	./run job-runner enqueue "$(FILE)" $(ARGS)

.PHONY: jobs-status
jobs-status:  ## Queue status from Redis. Usage: make jobs-status ARGS='--json'
	./run job-runner status $(ARGS)

.PHONY: jobs-result
jobs-result:  ## Get result for a request_id. Usage: make jobs-result ID=topic:beat-001
	@if [ -z "$(ID)" ]; then \
		echo "ERROR: ID is not set. Example: make jobs-result ID=topic:beat-001"; \
		exit 1; \
	fi
	./run job-runner result "$(ID)" $(ARGS)

.PHONY: jobs-report
jobs-report:  ## Status of all jobs in YAML. Usage: make jobs-report FILE=/app/jobs.example.yaml
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE is not set. Example: make jobs-report FILE=/app/jobs.example.yaml"; \
		exit 1; \
	fi
	./run job-runner report "$(FILE)" $(ARGS)

.PHONY: jobs-retry
jobs-retry:  ## Requeue a failed job. Usage: make jobs-retry ID=topic:beat-001 ARGS='--yes'
	@if [ -z "$(ID)" ]; then \
		echo "ERROR: ID is not set. Example: make jobs-retry ID=topic:beat-001"; \
		exit 1; \
	fi
	./run job-runner retry "$(ID)" $(ARGS)

.PHONY: jobs-flush
jobs-flush:  ## Flush transient queue keys. Add ARGS='--hard --yes' to also clear results.
	./run job-runner flush $(ARGS)

.PHONY: jobs-history
jobs-history:  ## Recent done/failed job history. Usage: make jobs-history ARGS='--limit 20 --status all'
	./run job-runner history $(ARGS)

.PHONY: jobs-dashboard
jobs-dashboard:  ## Quick queue dashboard: status + recent history in one shot
	@echo "── Queue Status ──────────────────────────────────────────────"
	@./run job-runner status 2>/dev/null || echo "(Redis not reachable)"
	@echo ""
	@echo "── Recent History (last 10) ──────────────────────────────────"
	@./run job-runner history --limit 10 2>/dev/null || echo "(Redis not reachable)"

.PHONY: jobs-logs
jobs-logs:  ## Tail the most recent execution logs. Usage: make jobs-logs [DIR=/work/job-batch]
	@DIR="$${DIR:-/work}"; \
	LATEST=$$(find "$$DIR" -name "execution.json" -type f 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No execution.json found under $$DIR"; \
		exit 1; \
	fi; \
	echo "Latest execution: $$LATEST"; \
	cat "$$LATEST" | python3 -m json.tool 2>/dev/null || cat "$$LATEST"

.PHONY: watcher-once
watcher-once:  ## Run one watcher scheduling cycle in the watcher compose service
	docker compose -f docker-compose.watcher.yml run --rm watcher python3 apps/job-watcher/job-watcher.py --once $(ARGS)

.PHONY: watcher-up
watcher-up:  ## Start the watcher daemon (docker-compose.watcher.yml)
	docker compose -f docker-compose.watcher.yml up -d watcher

.PHONY: watcher-down
watcher-down:  ## Stop the watcher daemon
	docker compose -f docker-compose.watcher.yml down

.PHONY: watcher-logs
watcher-logs:  ## Tail watcher daemon logs
	docker compose -f docker-compose.watcher.yml logs -f watcher

.PHONY: watcher-restart
watcher-restart:  ## Restart the watcher daemon (applies config/env changes)
	docker compose -f docker-compose.watcher.yml restart watcher

.PHONY: test
test:  ## Run the unit/integration test suite inside the toolbox container
	docker compose run --rm toolbox python -m pytest tests/ -v $(ARGS)

.PHONY: test-jobqueue-redis
test-jobqueue-redis:  ## Run opt-in Redis integration tests for lib/jobqueue.py (spins temporary redis)
	@docker compose run --rm toolbox python -c "import redis" >/dev/null 2>&1 || { \
		echo "ERROR: redis package is missing in toolbox image. Rebuild first: make build"; \
		exit 1; \
	}
	@set -eu; \
	REDIS_CONTAINER="$${REDIS_CONTAINER:-pab-redis-integration}"; \
	REDIS_IMAGE="$${REDIS_IMAGE:-redis:7-alpine}"; \
	REDIS_PORT="$${REDIS_PORT:-6380}"; \
	docker rm -f "$$REDIS_CONTAINER" >/dev/null 2>&1 || true; \
	trap 'docker rm -f "$$REDIS_CONTAINER" >/dev/null 2>&1 || true' EXIT; \
	docker run -d --name "$$REDIS_CONTAINER" -p "$$REDIS_PORT:6379" "$$REDIS_IMAGE" >/dev/null; \
	i=0; \
	while [ $$i -lt 40 ]; do \
		if docker exec "$$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then \
			break; \
		fi; \
		i=$$((i + 1)); \
		sleep 0.25; \
	done; \
	if ! docker exec "$$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then \
		echo "ERROR: Redis container did not become ready."; \
		exit 1; \
	fi; \
	REDIS_URL="$${PAB_TEST_REDIS_URL:-redis://host.docker.internal:$$REDIS_PORT/15}"; \
	echo "Running Redis integration tests against $$REDIS_URL"; \
	docker compose run --rm \
		-e PAB_RUN_REDIS_INTEGRATION=1 \
		-e PAB_TEST_REDIS_URL="$$REDIS_URL" \
		toolbox python -m pytest -q tests/integration/test_jobqueue_redis_integration.py

.PHONY: smoke-matrix
smoke-matrix:  ## Phase-5 smoke matrix (capabilities + clone + built-in + designed). Set SMOKE_SKIP_DESIGN=1 to skip design.
	./scripts/smoke-matrix.sh

.PHONY: synth-test
synth-test:  ## Comprehensive local CPU test matrix: CLI, voice-clone, clone_prompt, designed_clone, chunking, text-file, CustomVoice, tones, register-builtin, export/import, design-voice. Set SKIP_SLOW=1 or SKIP_DESIGN=1 to gate slow sections.
	./scripts/local-synth-test.sh $(ARGS)

.PHONY: test-local
test-local:  ## Full modular local test suite (tests/local/run-all.sh). Env: SKIP_SLOW=1, SKIP_DESIGN=1, ONLY="01 02", SKIP="18 19", TEST_AUDIO=/work/file.wav
	./tests/local/run-all.sh $(ARGS)

.PHONY: test-local-fast
test-local-fast:  ## Fast local suite: SKIP_SLOW=1 SKIP_DESIGN=1 (no variants, no design-voice, no slow FR clones)
	SKIP_SLOW=1 SKIP_DESIGN=1 ./tests/local/run-all.sh $(ARGS)

.PHONY: test-local-cli
test-local-cli:  ## CLI utilities only: list-voices, list-speakers, capabilities (01)
	bash tests/local/01-cli-utils.sh $(ARGS)

.PHONY: test-local-synth
test-local-synth:  ## Synthesis suites: clone profiles, designed, variants, chunking, text-file, save-profile, export/import (03-09)
	ONLY="03 04 05 06 07 08 09" ./tests/local/run-all.sh $(ARGS)

.PHONY: test-local-cv
test-local-cv:  ## CustomVoice suites: direct speaker, instruct, styles, register-builtin, tones, variants (10-15)
	ONLY="10 11 12 13 14 15" ./tests/local/run-all.sh $(ARGS)

# ── help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
