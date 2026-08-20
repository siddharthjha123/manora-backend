# MANORA Backend Observability Guide

Production-grade observability architecture for the MANORA Digital Mental Health and Psychological Support System backend.

---

## 1. Observability Architecture

```text
FastAPI Backend (Port 8000)
    │
    ├── Sentry SDK
    │   └── Unhandled exceptions, 5xx failures, and service errors (PII masked)
    │
    ├── Prometheus Instrumentation
    │   ├── Exposes GET /metrics
    │   ├── HTTP request duration, status codes, and traffic rates
    │   └── MANORA operational counters and latency histograms
    │
    ├── Langfuse AI Tracing
    │   └── Multi-stage student interaction traces & LLM generation metrics
    │
    └── Monitoring Stack (Docker Compose)
        ├── Prometheus (Port 9090) — Scrapes host.docker.internal:8000/metrics
        └── Grafana (Port 3000) — Visualizes system, LLM, and application metrics
```

```text
FastAPI
    ↓
Sentry       Prometheus       Langfuse
    │             ↓              │
Error Alerts   Grafana      LLM Dashboard
```

---

## 2. Observability Module Layout

```text
observability/
├── __init__.py         # Public exports (init_sentry, init_prometheus, init_langfuse, etc.)
├── sentry.py           # Sentry error monitoring & data scrubbing
├── langfuse.py         # Langfuse AI tracing, generations & spans
├── prometheus.py       # FastAPI HTTP instrumentation & /metrics mounting
├── metrics.py          # MANORA domain Prometheus metrics (counters & histograms)
└── middleware.py       # Observability middleware (X-Request-ID, duration headers)
```

---

## 3. Environment Variables

Add the following variables to your `.env` file:

```env
# Application Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Sentry Exception Monitoring (leave empty to disable in local dev)
SENTRY_DSN=
SENTRY_TRACES_SAMPLE_RATE=1.0

# Langfuse AI Observability (leave empty to disable in local dev)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Prometheus Application Metrics
PROMETHEUS_ENABLED=true
```

> [!NOTE]
> When `SENTRY_DSN` or `LANGFUSE_*` credentials are not configured, the backend automatically runs in graceful fallback mode with zero runtime errors or performance penalties.

---

## 4. Dependencies

The required observability dependencies are:

```text
sentry-sdk>=2.0.0
langfuse>=4.0.0
prometheus-fastapi-instrumentator>=8.0.0
prometheus_client>=0.20.0
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 5. Starting Prometheus and Grafana

To start the Prometheus and Grafana monitoring stack locally with Docker Compose:

```bash
docker-compose up -d
```

### Checking Running Services:

```bash
docker-compose ps
```

| Service             | Container Name        | Host Port             | Target URL              |
| :---                | :---                  | :---                  | :---                    |
| **Prometheus**      | `manora_prometheus`   | `9090`                | `http://localhost:9090` |
| **Grafana**         | `manora_grafana`      | `3000`                | `http://localhost:3000` |

---

## 6. Accessing Grafana & MANORA Dashboard

1. Open your browser and navigate to **`http://localhost:3000`**.
2. Log in with the default credentials:
   - **Username**: `admin`
   - **Password**: `admin`
3. Navigate to **Dashboards** > **MANORA** > **MANORA System & Observability Dashboard**.
4. The dashboard is pre-provisioned with four distinct observation sections:
   - **API Performance & HTTP Traffic**: QPS, 5xx Error Rate, p50/p95 latency, Status Code breakdown.
   - **LLM Performance & Inference**: Request Rate by Model, Error Rate by Type, p95 Latency.
   - **Memory Engine & Context Retrieval**: Retrieval Operations Rate, Retrieval Latency.
   - **MANORA Business Metrics**: Total Interactions, Emotion Predictions by Emotion, Buddy State Updates.

---

## 7. Configuring Prometheus

Prometheus is pre-configured via `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "manora-backend"
    metrics_path: "/metrics"
    scrape_interval: 15s
    static_configs:
      - targets: ["host.docker.internal:8000"]
        labels:
          app: "manora-backend"
          environment: "development"
```

To verify the scrape target in Prometheus:
1. Open **`http://localhost:9090/targets`**.
2. Verify `manora-backend` shows state **`UP`**.

---

## 8. Configuring Langfuse

1. Create a project at [cloud.langfuse.com](https://cloud.langfuse.com) (or self-hosted instance).
2. Generate API credentials in **Settings** > **API Keys**.
3. Set in `.env`:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```
4. Start your FastAPI server. Every interaction at `POST /interactions` will generate an end-to-end interaction trace containing:
   - Student Interaction Trace (`student_interaction`)
   - Emotion Agent ML + LLM generation
   - Buddy Agent LLM generation
   - Token consumption, latency, and model metadata.

---

## 9. Configuring Sentry

1. Create a project in [sentry.io](https://sentry.io).
2. Set your DSN in `.env`:
   ```env
   SENTRY_DSN=https://<key>@<org>.ingest.sentry.io/<project>
   ENVIRONMENT=development
   ```
3. All unhandled exceptions, FastAPI 500 errors, and logged `logger.error` events will automatically be captured in Sentry.
4. Authorization headers and sensitive student credentials are automatically redacted before sending via `_filter_sensitive_data`.

---

## 10. Testing `/metrics`

Test the metrics endpoint locally using `curl`:

```bash
curl http://localhost:8000/metrics
```

Expected response includes standard Prometheus metric lines:

```text
# HELP manora_interactions_total Total count of processed student interactions
# TYPE manora_interactions_total counter
manora_interactions_total{status="success"} 5.0

# HELP manora_emotion_predictions_total Total emotion predictions performed
# TYPE manora_emotion_predictions_total counter
manora_emotion_predictions_total{primary_emotion="anxiety"} 3.0
manora_emotion_predictions_total{primary_emotion="frustration"} 2.0

# HELP manora_llm_requests_total Total LLM generation requests
# TYPE manora_llm_requests_total counter
manora_llm_requests_total{model="openai/gpt-4o-mini",status="success"} 10.0
```

---

## 11. Security and Privacy Guidelines

1. **No High-Cardinality Labels**:
   - `user_id`, `session_id`, `interaction_id`, student text, and raw prompt content are **never** included as Prometheus metric labels.
2. **PII and Secret Redaction**:
   - `send_default_pii=False` is enforced in Sentry.
   - Authorization headers, cookies, and tokens are scrubbed.
3. **Graceful Failures**:
   - Observability failures (network outages, Langfuse downtime, Prometheus collection delays) **never** interrupt or degrade the student conversational experience.

---

## 12. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **Prometheus target DOWN** | `host.docker.internal` unreachable | Ensure Docker Desktop is running. If on Linux without host-gateway, use the bridge IP or run FastAPI in Docker network. |
| **Grafana datasource error** | Prometheus not reachable | Check `docker-compose ps` to ensure Prometheus container is running on port 9090. |
| **Langfuse traces not appearing** | Missing or invalid API keys | Check that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are populated in `.env`. |
| **Sentry events not sending** | `SENTRY_DSN` is empty | Sentry is disabled by design when DSN is empty. Populate `SENTRY_DSN` to activate. |
