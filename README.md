# MetricMedic Chatbot

MetricMedic is a lightweight retrieval-based chatbot designed for students and junior data analysts who encounter inconsistent metrics across SQL queries, dashboards, and business reports.

It uses TF-IDF vectorization and cosine similarity to match a user's question with the most relevant troubleshooting scenario and returns practical diagnostic steps through an interactive Streamlit chat interface.

## Features

- Troubleshoots SQL, dashboard, KPI, and data-quality issues
- Supports common problems such as:
  - duplicate records
  - incorrect joins
  - NULL handling
  - filter mismatches
  - stale dashboard data
  - revenue discrepancies
  - KPI definition differences
- Retrieval-based NLP using TF-IDF and cosine similarity
- JSON troubleshooting knowledge base
- Interactive Streamlit chat interface
- Command-line interface
- Low-confidence fallback for unrelated questions
- Automated tests with pytest
- No API key required

## Tech Stack

- Python
- scikit-learn
- Streamlit
- pytest
- JSON

## Project Structure

```text
metricmedic-chatbot/
├── app.py
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

Before running MetricMedic, make sure you have:

- Python 3.10 or later
- pip
- Git

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

## Run the Streamlit Chatbot

Start the interactive web interface with:

```bash
streamlit run app.py
```

Streamlit will usually open the application in your browser at:

```text
http://localhost:8501
```

Example question:

```text
My dashboard customer count doubled after I joined two tables.
```

Example response:

```text
Possible issue: COUNT and COUNT DISTINCT mismatch

Recommended checks:
Compare COUNT(*) with COUNT(DISTINCT identifier).
A difference may indicate duplicate rows or one-to-many joins.
```

## Run the Command-Line Version

MetricMedic can also be used directly from the terminal:

```bash
python chatbot.py
```

Example:

```text
You: My Power BI dashboard shows more revenue than my SQL query

MetricMedic:
Possible issue: Power BI dashboard mismatch

Recommended checks:
Review Power BI filters, relationships, DAX measures, and aggregation context.
Compare the underlying rows with the source SQL.
```

Type `exit`, `quit`, or `bye` to close the chatbot.

## How It Works

MetricMedic follows a lightweight retrieval-based NLP workflow:

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

Troubleshooting scenarios are stored in:

```text
data/troubleshooting_guide.json
```

The chatbot converts the stored troubleshooting scenarios and the user's question into TF-IDF vectors using scikit-learn.

Cosine similarity compares the user's question against the knowledge base and retrieves the closest matching troubleshooting issue.

If the similarity score is too low, MetricMedic returns a fallback response instead of presenting an unrelated recommendation.

## Knowledge Base

The JSON knowledge base contains troubleshooting scenarios related to:

- duplicate records
- incorrect SQL joins
- NULL values
- COUNT vs. COUNT DISTINCT
- date-filter mismatches
- stale dashboard data
- revenue mismatches
- Power BI reporting issues
- KPI definition inconsistencies

The knowledge base can be expanded by adding additional entries to:

```text
data/troubleshooting_guide.json
```

## Testing

Run the automated tests with:

```bash
pytest
```

The test suite checks:

- chatbot initialization
- knowledge-base loading
- Power BI issue matching
- empty input handling
- unrelated-question fallback behavior

Expected result:

```text
4 passed
```

## Dependencies

The project uses:

```text
scikit-learn==1.5.1
pytest==9.0.3
streamlit
```

- **scikit-learn** — TF-IDF vectorization and cosine similarity
- **pytest** — automated testing
- **Streamlit** — browser-based chatbot interface

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.