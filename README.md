# MetricMedic Chatbot

MetricMedic is a lightweight troubleshooting chatbot designed for students and junior data analysts who encounter inconsistent metrics across SQL queries, dashboards, and business reports.

It uses TF-IDF vectorization and cosine similarity to match a user's question with the most relevant troubleshooting scenario and returns practical diagnostic steps.

## Features

- Troubleshoots dashboard and SQL mismatches
- Detects likely issues involving:
  - duplicate records
  - incorrect joins
  - NULL values
  - filter mismatches
  - stale dashboard data
  - KPI definition differences
  - Power BI reporting issues
- Retrieval-based NLP
- Lightweight JSON knowledge base
- No API key required
- Includes automated tests with pytest

## Tech Stack

- Python 3
- scikit-learn
- pytest
- JSON

## Project Structure

```text
metricmedic-chatbot/
├── chatbot.py
├── data/
│   └── troubleshooting_guide.json
├── tests/
│   └── test_chatbot.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE