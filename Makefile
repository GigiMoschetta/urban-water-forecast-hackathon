.PHONY: verify reproduce reproduce-legacy demo build clean help

# Auto-detect container runtime: prefer docker compose plugin, then podman-compose
COMPOSE := $(or \
  $(shell command -v podman-compose 2>/dev/null), \
  $(shell docker compose version >/dev/null 2>&1 && echo "docker compose"), \
  $(shell command -v docker-compose 2>/dev/null))

ifeq ($(COMPOSE),)
  $(warning No container runtime found. Install Docker or Podman + podman-compose.)
endif

help: ## Show available commands
	@echo ""
	@echo "  hackathon-opti — Water Demand Forecasting"
	@echo "  ─────────────────────────────────────────"
	@echo ""
	@echo "  make verify            Verify submission artifact integrity"
	@echo "  make reproduce         Rebuild 3-model ensemble (~10 min)"
	@echo "  make reproduce-legacy  Rebuild historical 7-model path (~10 min)"
	@echo "  make demo              Launch interactive demo (http://localhost:3000)"
	@echo "  make build             Build container image"
	@echo "  make clean             Remove containers and images"
	@echo ""
	@echo "  Detected runtime: $(COMPOSE)"
	@echo ""

build: ## Build container image
	$(COMPOSE) build

verify: build ## Verify submission artifact
	$(COMPOSE) run --rm verify

reproduce: build ## Rebuild 3-model ensemble (naive+v1+v9o)
	$(COMPOSE) --profile reproduce run --rm reproduce

reproduce-legacy: build ## Rebuild historical 7-model path
	$(COMPOSE) --profile legacy run --rm reproduce-legacy

demo: ## Launch interactive demo dashboard (http://localhost:3000)
	$(COMPOSE) --profile demo up --build demo

clean: ## Remove containers and images
	$(COMPOSE) down --rmi local --remove-orphans
