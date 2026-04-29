# AI Anomaly Detection & Explanation System

> A hybrid AI system combining a TensorFlow time-series anomaly detection model with an LLM agent that automatically generates plain-language root cause analysis reports — deployed as a Grafana-integrated monitoring dashboard with bilingual (EN/DE) output.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/AbdelrahmanMohamed54/Anomaly-Detection-Explanation-System
cd Anomaly-Detection-Explanation-System
cp .env.example .env   # add your GOOGLE_API_KEY or GROQ_API_KEY

# 2. Start all services with Docker
docker-compose up --build

# 3. Run one-time data setup (inside the running container)
docker-compose exec api python scripts/generate_synthetic_data.py
docker-compose exec api python model/train.py
docker-compose exec api python scripts/seed_postgres.py
docker-compose exec api python scripts/ingest_incident_reports.py
docker-compose exec api python scripts/ingest_maintenance_logs.py

# 4. Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"sensor_id":"PUMP_01","anomaly_type":"bearing_wear","severity":0.85,"language":"both"}'

# 5. Open Grafana dashboard
# Navigate to http://localhost:3000 (admin/admin), import grafana/dashboard.json
```

---

## Demo Output

The following is real output from `python scripts/run_demo.py` — PUMP_01 bearing wear scenario:

```
========================================================================
  SCENARIO 1 REPORT — EN
========================================================================
  Sensor:   PUMP_01
  Severity: HIGH  (0.85)
  Escalate: YES — immediate action required
------------------------------------------------------------------------
  ANOMALY SUMMARY
------------------------------------------------------------------------
  An anomaly has been detected on PUMP_01, classified as bearing wear.
  The pump is exhibiting elevated temperature (96.0 C) and high vibration
  (3.8 mm/s), while operating at 1495 RPM with a current draw of 13.5 A
  and pressure of 5.0 bar. The severity score is 0.850.

------------------------------------------------------------------------
  ROOT CAUSE
------------------------------------------------------------------------
  The most probable root cause is severe wear of the drive-end bearing on
  PUMP_01, leading to increased friction, heat generation, and vibration.
  This is supported by historical incident data showing identical symptoms
  on PUMP_01's drive-end bearing.

------------------------------------------------------------------------
  SIMILAR INCIDENTS
------------------------------------------------------------------------
  1. PUMP_01 experienced a high-vibration alarm (3.8 mm/s) and bearing
     temperature of 96 C on its drive-end bearing, leading to manual
     isolation. (Source: incident_search_tool)
  2. MOTOR_01's non-drive-end bearing temperature gradually increased to
     102 C with vibration reaching 2.9 mm/s. (Source: incident_search_tool)
  3. PUMP_02 vibration survey identified 2x running speed harmonic and
     12 C temperature rise on the coupling-side bearing.

------------------------------------------------------------------------
  RECOMMENDED ACTIONS
------------------------------------------------------------------------
  1. Immediately isolate PUMP_01 to prevent secondary damage.
  2. Perform an emergency replacement of the drive-end bearing on PUMP_01.
  3. Inspect the removed bearing for failure modes such as spalling or
     abrasive wear.

------------------------------------------------------------------------
  Sources: historical_anomaly_tool, incident_search_tool, corrective_action_tool
  Generated in 29.5s
========================================================================
```

---

## Test Results

```
$ pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-9.0.3
collected 64 items

tests/test_agent.py::TestRCAAgent::test_rca_report_has_all_fields_populated PASSED
tests/test_agent.py::TestRCAAgent::test_escalate_true_when_severity_above_threshold PASSED
tests/test_agent.py::TestRCAAgent::test_escalate_true_when_severity_well_above_threshold PASSED
tests/test_agent.py::TestRCAAgent::test_escalate_false_for_low_severity PASSED
tests/test_agent.py::TestRCAAgent::test_llm_escalate_true_overrides_low_severity PASSED
tests/test_agent.py::TestRCAAgent::test_generate_report_called PASSED
tests/test_agent.py::TestRCAAgent::test_similar_incidents_capped_at_three PASSED
tests/test_agent.py::TestReportGenerator::test_en_report_has_all_required_fields PASSED
tests/test_agent.py::TestReportGenerator::test_bilingual_reports_have_different_text PASSED
tests/test_agent.py::TestReportGenerator::test_de_report_contains_german_text PASSED
tests/test_agent.py::TestSeverityLabel::test_high_severity_mapping PASSED
tests/test_agent.py::TestSeverityLabel::test_medium_severity_mapping PASSED
tests/test_agent.py::TestSeverityLabel::test_low_severity_mapping PASSED
tests/test_agent.py::TestSeverityLabel::test_boundary_high_threshold PASSED
tests/test_agent.py::TestSeverityLabel::test_boundary_medium_threshold PASSED
tests/test_agent.py::TestSeverityLabel::test_german_high_label PASSED
tests/test_agent.py::TestSeverityLabel::test_german_medium_label PASSED
tests/test_agent.py::TestSeverityLabel::test_german_low_label PASSED
tests/test_agent.py::TestFullRCAAgentPipeline::test_full_rca_agent_returns_valid_report PASSED
tests/test_agent.py::TestFullRCAAgentPipeline::test_full_rca_agent_all_tools_referenced_in_sources PASSED
tests/test_api.py (14 tests) ................  PASSED
tests/test_model.py (12 tests) ............  PASSED
tests/test_tools.py (18 tests) ..................  PASSED

======================== 64 passed in 19.21s ==================================
```

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

1. Install the JSON API plugin (if not already installed):
   ```bash
   grafana-cli plugins install marcusolsson-json-datasource
   ```
2. Go to **Configuration > Data Sources > Add data source**, search **JSON API**
3. Set URL to `http://localhost:8000`, name it `Anomaly Detection API`, click **Save & Test**
4. Go to **Dashboards > Import > Upload dashboard JSON file**
5. Select `grafana/dashboard.json` and map the datasource to `Anomaly Detection API`
6. Click **Import** — the dashboard auto-refreshes every 30 seconds

Populate the history panels by running a simulate call first:
```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"sensor_id":"PUMP_01","anomaly_type":"bearing_wear","severity":0.85,"language":"both"}'
```

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
[LinkedIn](https://linkedin.com/in/abdelrahman-25-mohamed) · [GitHub](https://github.com/AbdelrahmanMohamed54)  
abdelrahman.mohamed2505@gmail.com
