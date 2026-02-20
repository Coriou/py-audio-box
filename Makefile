# Toolbox Makefile — shortcuts for common tasks
# Usage: make <target> [ARGS="...extra flags..."]
#
# Most workflows go through ./run <app> instead, but this is handy for
# build / shell / one-liner invocations.

.DEFAULT_GOAL := help

# ── infrastructure ─────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build (or rebuild) the toolbox Docker image
	docker compose build

.PHONY: build-no-cache
build-no-cache:  ## Force a clean rebuild (no layer cache)
	docker compose build --no-cache

.PHONY: shell
shell:  ## Open an interactive bash shell inside the toolbox
	docker compose run --rm toolbox bash

.PHONY: clean-work
clean-work:  ## Delete all output files in ./work/ (keeps cache)
	rm -rf work/clip_*.wav

.PHONY: clean-cache
clean-cache:  ## Wipe the shared cache (forces model re-download on next run)
	rm -rf cache/

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

# ── help ───────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
