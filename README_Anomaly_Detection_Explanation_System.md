# AI Anomaly Detection & Explanation System

> A hybrid AI system combining a TensorFlow time-series anomaly detection model with an LLM agent that automatically generates plain-language root cause analysis reports — deployed as a Grafana-integrated monitoring dashboard with bilingual (EN/DE) output.

---

## Overview

Industrial anomaly detection systems are good at flagging problems but terrible at explaining them. Plant operators receive an alert but still need to manually investigate root causes, consult historical records, and decide on corrective actions — a process that can take 30–60 minutes per incident.

This system closes that gap by pairing a **TensorFlow-based time-series anomaly detector** with a **LangChain LLM agent** that automatically generates structured root cause analysis reports the moment an anomaly is flagged. Reports are produced in under 30 seconds, in both English and German, and surfaced directly in a **Grafana monitoring dashboard**.

---

## Architecture

```
Manufacturing Sensor Data (time-series)
          │
          ▼
┌─────────────────────────────┐
│   TensorFlow Anomaly Model   │  ← LSTM / Autoencoder trained on historical data
│   (time-series classification)│
└─────────────┬───────────────┘
              │ anomaly detected + severity score
              ▼
┌─────────────────────────────┐
│     LangChain LLM Agent      │  ← Claude API — root cause analysis
│     (Root Cause Analyzer)    │
│                              │
│  Tools available:            │
│  ├── historical_query()      │  → PostgreSQL: similar past anomalies
│  ├── semantic_search()       │  → ChromaDB: incident report retrieval
│  └── action_recommender()    │  → Maintenance log RAG → corrective actions
└─────────────┬───────────────┘
              │ structured RCA report (EN + DE)
              ▼
┌─────────────────────────────┐
│        FastAPI Backend       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Grafana Dashboard         │  ← Real-time anomaly alerts + LLM report panel
│    (live monitoring)         │
└─────────────────────────────┘
```

---

## Key Features

- **Hybrid AI architecture** — classical ML anomaly detection feeds directly into an LLM reasoning agent, combining the precision of trained models with the explanatory power of LLMs
- **Three specialized agent tools:**
  - `historical_query` — queries PostgreSQL for past anomalies matching sensor type, timestamp range, and severity
  - `semantic_search` — searches ChromaDB for semantically similar incident reports using embedding-based retrieval
  - `action_recommender` — retrieves relevant entries from maintenance logs and recommends corrective actions grounded in past resolutions
- **Bilingual output** — all RCA reports generated in both English and German, enabling use across international manufacturing teams
- **Grafana integration** — anomaly alerts and LLM-generated explanations surface directly in the existing monitoring dashboard via FastAPI data source plugin
- **Mean time to diagnosis: ~30 seconds** vs ~45 minutes manual review in a simulated manufacturing environment

---

## Agent Tools Detail

### `historical_query(sensor_id, anomaly_type, time_range)`
Queries a PostgreSQL database of historical anomaly events. Returns the 5 most similar past incidents by sensor type and anomaly classification, including timestamps, durations, and recorded outcomes.

### `semantic_search(anomaly_description)`
Embeds the anomaly description and searches ChromaDB over a corpus of past incident reports. Returns the top-k semantically similar reports with their root cause findings and resolution notes.

### `action_recommender(root_cause)`
Retrieves relevant entries from a RAG-indexed maintenance log database. Given an identified root cause, recommends specific corrective actions drawn from documented past resolutions.

---

## RCA Report Structure

Each generated report contains:

```
ANOMALY REPORT — [Timestamp]
Sensor: [ID] | Severity: [HIGH/MEDIUM/LOW] | Confidence: [%]

1. Anomaly Summary
   [Plain-language description of what was detected]

2. Most Probable Root Cause
   [LLM reasoning grounded in retrieved historical data]

3. Similar Past Incidents
   [Top 3 historical matches with dates and outcomes]

4. Recommended Corrective Actions
   [Ordered action list from maintenance log RAG]

5. Escalation Recommendation
   [Whether human expert review is advised]

Sources: [List of retrieved documents used in this report]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Anomaly Detection Model | TensorFlow (LSTM / Autoencoder) |
| LLM Agent | LangChain + Claude API (Anthropic) |
| Historical Query Tool | PostgreSQL |
| Semantic Search Tool | ChromaDB + sentence-transformers |
| Maintenance Log RAG | ChromaDB + LangChain retriever |
| API Backend | FastAPI |
| Monitoring Dashboard | Grafana (custom data source plugin) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
anomaly-detection-llm/
├── model/
│   ├── train.py              # TensorFlow model training pipeline
│   ├── anomaly_detector.py   # Inference wrapper + threshold logic
│   └── data_preprocessing.py # Time-series feature engineering
├── agent/
│   ├── rca_agent.py          # LangChain agent + tool orchestration
│   ├── tools/
│   │   ├── historical_query.py    # PostgreSQL tool
│   │   ├── semantic_search.py     # ChromaDB tool
│   │   └── action_recommender.py  # Maintenance log RAG tool
│   └── report_generator.py   # Bilingual report formatting (EN/DE)
├── api/
│   └── main.py               # FastAPI backend + Grafana data source endpoint
├── grafana/
│   └── dashboard.json        # Grafana dashboard config (importable)
├── data/
│   ├── sensor_data/          # Sample time-series sensor data
│   ├── incident_reports/     # Historical incident report corpus
│   └── maintenance_logs/     # Maintenance log corpus
├── scripts/
│   ├── ingest_incident_reports.py   # Embed + store in ChromaDB
│   ├── ingest_maintenance_logs.py
│   └── seed_postgres.py             # Seed historical anomaly DB
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Anthropic API key
- PostgreSQL (included in docker-compose)
- Grafana (included in docker-compose)

### Installation

```bash
git clone https://github.com/AbdelrahmanMohamed54/anomaly-detection-llm
cd anomaly-detection-llm
cp .env.example .env
# Add: ANTHROPIC_API_KEY, POSTGRES_URL
docker-compose up --build
```

### Ingest knowledge base

```bash
python scripts/ingest_incident_reports.py
python scripts/ingest_maintenance_logs.py
python scripts/seed_postgres.py
```

### Train the anomaly detection model

```bash
python model/train.py --data data/sensor_data/ --epochs 50 --model lstm
```

### Access the system

| Service | URL |
|---|---|
| Grafana Dashboard | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

### Import Grafana dashboard

In Grafana → Dashboards → Import → Upload `grafana/dashboard.json`

---

## API Usage

### Trigger anomaly analysis manually

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "PUMP_03",
    "anomaly_type": "vibration_spike",
    "severity": 0.87,
    "timestamp": "2026-03-15T14:32:00Z",
    "language": "EN"
  }'
```

### Response

```json
{
  "report_id": "rca_20260315_143200",
  "sensor_id": "PUMP_03",
  "severity": "HIGH",
  "root_cause": "Bearing wear consistent with lubrication failure...",
  "similar_incidents": [...],
  "recommended_actions": [...],
  "escalate": true,
  "sources": [...],
  "generation_time_seconds": 28.4
}
```

---

## Background & Motivation

This project is inspired by real manufacturing data engineering work at HORSCH Maschinen SE & Co. KG (2025), where predictive TensorFlow models were deployed on production sensor data. This system extends that work by adding an LLM reasoning layer — bridging the gap between anomaly detection (what happened) and root cause analysis (why it happened and what to do).

---

## Author

**Abdelrahman Mohamed** — AI Engineer  
[LinkedIn](https://linkedin.com/in/abdelrahman-mohamed) · [GitHub](https://github.com/AbdelrahmanMohamed54)  
abdelrahman.mohamed2505@gmail.com
