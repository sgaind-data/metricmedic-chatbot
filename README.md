# MetricMedic Chatbot

MetricMedic is a lightweight troubleshooting chatbot designed for students and junior data analysts who encounter inconsistent metrics across SQL queries, dashboards, and business reports.

It uses TF-IDF vectorization and cosine similarity to match a user's question with the most relevant troubleshooting scenario and returns practical diagnostic steps.

## Features

* Troubleshoots dashboard and SQL mismatches
* Detects likely issues involving:

  * duplicate records
  * incorrect joins
  * NULL values
  * filter mismatches
  * stale dashboard data
  * KPI definition differences
  * Power BI reporting issues
* Retrieval-based NLP
* Lightweight JSON knowledge base
* No API key required
* Includes automated tests with pytest

## Tech Stack

* Python 3
* scikit-learn
* pytest
* JSON

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
```

## Prerequisites

* Python 3.10 or later
* pip

Check your Python version:

```bash
python3 --version
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sgaind-data/metricmedic-chatbot.git
cd metricmedic-chatbot
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Chatbot

Start MetricMedic with:

```bash
python chatbot.py
```

Example interaction:

```text
You: My Power BI dashboard shows more revenue than my SQL query

MetricMedic:
Possible issue: Power BI dashboard mismatch

Recommended checks:
Review Power BI filters, relationships, DAX measures, and aggregation context.
Compare the underlying rows with the source SQL.
```

Type `exit`, `quit`, or `bye` to close the chatbot.

## Run Tests

Run the automated test suite with:

```bash
pytest
```

The project currently includes tests for:

* chatbot initialization
* Power BI issue matching
* empty input handling
* unrelated questions

## How It Works

MetricMedic uses a lightweight retrieval-based NLP workflow:

```text
User Question
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Closest Troubleshooting Scenario
      ↓
Recommended Diagnostic Steps
```

Troubleshooting scenarios are stored in `data/troubleshooting_guide.json`.

The chatbot converts the stored scenarios and the user's question into TF-IDF vectors and uses cosine similarity to retrieve the closest matching issue.

## Dependencies

* `scikit-learn==1.5.1` — TF-IDF vectorization and cosine similarity
* `pytest==9.0.3` — automated testing

## Testing

All automated tests should pass with:

```bash
pytest
```

Expected result:

```text
4 passed
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
