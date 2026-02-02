.PHONY: dev run test lint format check clean install build push deploy undeploy setup-local

# Container image settings
IMAGE_REGISTRY ?= quay.io
IMAGE_ORG ?= afalossi
IMAGE_NAME ?= observability-assistant-router
IMAGE_TAG ?= latest
IMAGE ?= $(IMAGE_REGISTRY)/$(IMAGE_ORG)/$(IMAGE_NAME):$(IMAGE_TAG)

# Local development settings
LOCAL_PORT ?= 9010
LOCAL_CONFIG ?= config/agents.yaml

# Install dependencies
install:
	uv sync

# Setup local development environment
setup-local:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "Done. Edit .env to customize settings."; \
	else \
		echo ".env already exists. Skipping."; \
	fi

# Run the development server with hot reload (uses .env if present)
dev:
	PYTHONPATH=src ROUTER_CONFIG_PATH=$(LOCAL_CONFIG) uv run uvicorn router.main:app --reload --host 0.0.0.0 --port $(LOCAL_PORT)

# Run the production server (uses .env if present)
run:
	PYTHONPATH=src ROUTER_CONFIG_PATH=$(LOCAL_CONFIG) uv run uvicorn router.main:app --host 0.0.0.0 --port $(LOCAL_PORT)

# Run tests
test:
	PYTHONPATH=src uv run pytest tests/ -v

# Run linter
lint:
	uv run ruff check src/ tests/

# Format code
format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# Run all checks (lint + test)
check: lint test

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Container Build
# ============================================================================

# Build container image
build:
	podman build -t $(IMAGE) .

# Push container image to registry
push:
	podman push $(IMAGE)

# Build and push in one step
release: build push

# ============================================================================
# OpenShift Deployment
# ============================================================================

# Deploy to OpenShift (applies all manifests)
deploy:
	oc apply -f deploy/namespace.yaml
	oc apply -f deploy/serviceaccount.yaml
	oc apply -f deploy/configmap.yaml
	oc apply -f deploy/deployment.yaml
	oc apply -f deploy/service.yaml
	oc apply -f deploy/route.yaml

# Remove deployment from OpenShift
undeploy:
	oc delete -f deploy/route.yaml --ignore-not-found
	oc delete -f deploy/service.yaml --ignore-not-found
	oc delete -f deploy/deployment.yaml --ignore-not-found
	oc delete -f deploy/configmap.yaml --ignore-not-found
	oc delete -f deploy/serviceaccount.yaml --ignore-not-found
	oc delete -f deploy/namespace.yaml --ignore-not-found

# Show deployment status
status:
	@echo "=== Namespace ==="
	oc get namespace observability-assistant
	@echo ""
	@echo "=== Pods ==="
	oc get pods -l app=observability-assistant-router -n observability-assistant
	@echo ""
	@echo "=== Service ==="
	oc get svc observability-assistant-router -n observability-assistant
	@echo ""
	@echo "=== Route ==="
	oc get route observability-assistant-router -n observability-assistant

# Stream logs from the router pod
logs:
	oc logs -f -l app=observability-assistant-router -n observability-assistant
