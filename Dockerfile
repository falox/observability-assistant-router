# Multi-stage Dockerfile for observability-assistant-router
# Uses uv for fast, reliable Python package management

FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ ./src/

# Install the project
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# Create non-root user (OpenShift requirement: runs as arbitrary UID)
RUN useradd -m -u 1001 router && \
    mkdir -p /config && \
    chown -R 1001:0 /config && \
    chmod -R g=u /config

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --from=builder /app/src ./src

# Copy default config (will be overridden by ConfigMap mount)
COPY config/ ./config/

# Set ownership for OpenShift (group 0 with same permissions as owner)
RUN chown -R 1001:0 /app && chmod -R g=u /app

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER 1001

EXPOSE 9010

CMD ["uvicorn", "router.main:app", "--host", "0.0.0.0", "--port", "9010"]
