# observability-assistant-router

A multi-agent orchestrator service that routes messages to specialized agents via semantic matching and streams responses back to clients using the AG-UI protocol.

## Features

- **Semantic Routing**: Routes messages to agents based on semantic similarity using sentence-transformers
- **Multi-Protocol Support**: Supports both AG-UI and A2A (Agent-to-Agent) protocols
- **Sticky Sessions**: Maintains conversation affinity with agents across messages
- **Topic Drift Detection**: Automatically re-routes when conversation topic changes
- **@Mention Override**: Users can force routing to specific agents with `@handle` syntax
- **LLM Fallback**: Uses default agent's LLM for ambiguous routing decisions
- **BYOA (Bring Your Own Agent)**: Register agents via YAML configuration - no code changes required

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              observability-assistant-router                     │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │ AG-UI    │───>│   Semantic   │───>│  Agent Proxy    │        │
│  │ Endpoint │    │   Router     │    │                 │        │
│  │ (SSE)    │<───│              │<───│                 │        │
│  └──────────┘    └──────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Default     │  │  Agent A    │  │  Agent B    │
  │ Agent       │  │  (A2A)      │  │  (AG-UI)    │
  └─────────────┘  └─────────────┘  └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/observability-assistant-router.git
cd observability-assistant-router

# Install dependencies
make install
```

### Running Locally

```bash
# Start development server with hot reload
make dev

# Or run production server
make run
```

The server starts at `http://localhost:9010`.

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:9010/health/live

# Send a test request
curl -X POST http://localhost:9010/api/agui/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-123",
    "messages": [{"id": "msg-1", "role": "user", "content": "Why is my pod crashing?"}]
  }'
```

## Configuration

Agents are configured via YAML. See `config/agents.yaml`:

```yaml
session:
  sticky_enabled: true  # Keep using the same agent during a conversation
  timeout_minutes: 30  # How long before a conversation expires
  topic_drift_threshold: 0.2  # How different a message must be to switch agents (0-1, lower = stickier)

default_agent:
  id: "default"

agents:
  - id: "default"
    name: "Generic"
    handles: ["assistant", "generic"]
    url: "http://localhost:8080/a2a"
    protocol: "a2a"
    description: >
      General-purpose assistant (fallback agent).

  - id: "troubleshooting-agent"
    name: "Troubleshooting"
    handles: ["issue", "alert", "debug"]
    url: "http://localhost:5050/api/agui/chat"
    protocol: "ag-ui"

    routing:
      priority: 1  # Lower = checked first when multiple agents match
      threshold: 0.80  # How confident the match must be (0-1, higher = stricter)
      examples:
        - "Why is my pod crashing?"
        - "What triggered the high CPU alert?"
        - "Debug DNS resolution failures"
        # ...

    description: >
      Investigates cluster issues including alerts and root cause analysis.
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_HOST` | `0.0.0.0` | Server bind address |
| `ROUTER_PORT` | `9010` | Server port |
| `ROUTER_LOG_LEVEL` | `INFO` | Logging level |
| `ROUTER_CONFIG_PATH` | `/config/agents.yaml` | Path to agents config |
| `ROUTER_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `ROUTER_SESSION_ENABLED` | `true` | Enable sticky sessions |
| `ROUTER_RETRY_ATTEMPTS` | `2` | Max retry attempts for failed requests |
| `ROUTER_RETRY_BACKOFF_MS` | `500` | Base delay (ms) for exponential backoff |
| `ROUTER_AUDIT_ENABLED` | `true` | Enable structured audit logging |

## Development

### Commands

```bash
make install    # Install dependencies
make dev        # Run dev server with hot reload
make run        # Run production server
make test       # Run tests
make lint       # Check code style
make format     # Format code
make check      # Run lint + test
make clean      # Remove cache files
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_routing.py -v

# Run with coverage
uv run pytest tests/ --cov=src/router
```

### Project Structure

```
src/router/
├── main.py           # FastAPI app entry point
├── config/           # Settings and agent configuration
├── routing/          # Semantic router, @mention, drift detection
├── session/          # Sticky session store
├── agui/             # AG-UI protocol endpoint and client
├── a2a/              # A2A protocol client and translator
├── agents/           # Agent proxy for forwarding requests
└── observability/    # Audit logging and stream buffering

tests/                # Test suite
config/agents.yaml    # Example agent configuration
```

## Container Build

```bash
# Build container image (uses podman by default)
make build

# Push to registry
make push

# Build and push
make release

# Customize image location
make build IMAGE_REGISTRY=quay.io IMAGE_ORG=myorg IMAGE_TAG=v1.0.0
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/agui/chat` | AG-UI chat endpoint (SSE streaming) |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/ready` | Kubernetes readiness probe |
| `GET` | `/` | API metadata |

### Chat Request Format

```json
{
  "thread_id": "conversation-123",
  "messages": [
    {"id": "msg-1", "role": "user", "content": "Why is my pod crashing?"}
  ]
}
```

### Using @Mentions

Force routing to a specific agent:

```
@troubleshoot why is my pod crashing?
@metrics show CPU usage
@assistant help me with something general
```

- Case-insensitive: `@Metrics` and `@metrics` both work
- First mention wins: `@metrics @troubleshoot help` routes to metrics

## Routing Logic

1. **@Mention Check**: If message contains `@handle`, route to that agent
2. **Session Check**: If sticky session exists, check for topic drift
3. **Semantic Matching**: Embed message and compare to agent examples
4. **LLM Fallback**: If no match above threshold, ask default agent's LLM
5. **Default Fallback**: If still uncertain, route to default agent
