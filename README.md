# MetricMedic 🩺📊

MetricMedic is a RAG-powered troubleshooting assistant designed for students and junior data analysts who encounter mismatched metrics across SQL queries, dashboards, ETL pipelines, KPIs, and business reports.

It combines semantic embeddings, FAISS vector search, a structured analytics knowledge base, and Gemini to retrieve relevant troubleshooting context and generate grounded diagnostic guidance through an interactive Streamlit chat interface.

## Features

- Natural-language troubleshooting for SQL, dashboards, KPIs, ETL, and data-quality issues
- Semantic retrieval using Sentence Transformers
- FAISS vector similarity search
- Gemini-powered grounded response generation
- Structured JSON troubleshooting knowledge base
- Retrieval transparency with similarity scores
- Persistent chat history using SQLite
- New Chat and previous-chat navigation
- Automatic chat titles
- Delete saved conversations
- Fallback response if Gemini is temporarily unavailable
- Automated retrieval tests with pytest
- Streamlit conversational interface

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Google Gemini
- SQLite
- JSON
- pytest

## Architecture

```text
User Question
      ↓
SentenceTransformer Embedding
      ↓
FAISS Vector Search
      ↓
Top-K Troubleshooting Context
      ↓
Grounded Prompt
      ↓
Gemini
      ↓
Generated Troubleshooting Answer
```

If Gemini is temporarily unavailable, MetricMedic falls back to a deterministic response based on the retrieved troubleshooting context.

## Project Structure

```text
metricmedic-chatbot/
├── app.py
├── chat_store.py
├── llm.py
├── rag.py
├── rag_chatbot.py
├── data/
│   └── troubleshooting_guide.json
├── tests/
│   └── test_rag.py
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How It Works

### 1. Semantic Retrieval

MetricMedic loads troubleshooting scenarios from:

```text
data/troubleshooting_guide.json
```

Each scenario contains an issue description, keywords, and recommended diagnostic guidance.

The content is converted into semantic embeddings using a Sentence Transformer model.

### 2. FAISS Vector Search

The user's question is embedded using the same model.

FAISS compares the query embedding with the troubleshooting knowledge base and retrieves the most relevant scenarios using vector similarity.

### 3. Retrieval-Augmented Generation

The top retrieved scenarios are passed to Gemini as grounded context.

Gemini generates a concise response containing:

- the likely issue
- recommended diagnostic checks
- an explanation of why the issue may be happening

The model is instructed to rely on the retrieved troubleshooting context rather than inventing unsupported details.

### 4. Persistent Chat History

MetricMedic stores conversations locally using SQLite.

Users can:

- start a new chat
- reopen previous conversations
- maintain chat history between sessions
- automatically title conversations from the first question
- delete saved chats

The local database is excluded from GitHub.

## Prerequisites

Before running MetricMedic, make sure you have:

- Python 3.10 or later
- pip
- Git
- Gemini API key

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

## Configure Gemini

A template environment file is included:

```text
.env.example
```

Create your own `.env` file in the project root:

```text
GEMINI_API_KEY=your_actual_gemini_api_key
```

Do not commit your `.env` file or API key to GitHub.

## Run MetricMedic

Start the Streamlit application:

```bash
streamlit run app.py
```

The application usually opens at:

```text
http://localhost:8501
```

## Example Questions

Try prompts such as:

```text
My dashboard customer count doubled after joining two tables.
```

```text
Finance says revenue is different from Power BI.
```

```text
My dashboard is still showing yesterday's data.
```

```text
Why did records disappear after an INNER JOIN?
```

```text
My conversion rate is above 100%.
```

```text
My historical dashboard numbers changed from last month.
```

## Example Retrieval Flow

For a question such as:

```text
My dashboard customer count doubled after joining two tables.
```

MetricMedic may retrieve scenarios related to:

```text
Dashboard total is higher than the source system
COUNT and COUNT DISTINCT mismatch
Many-to-many join causing row explosion
```

Gemini then combines the retrieved troubleshooting context into a grounded response.

Users can expand the **Retrieved troubleshooting context** section in the interface to inspect the supporting issues and similarity scores.

## Testing

Run the automated test suite with:

```bash
pytest
```

The tests validate:

- knowledge-base loading
- FAISS index creation
- semantic retrieval
- join-problem retrieval
- stale-dashboard retrieval
- revenue-mismatch retrieval

Gemini is not called during unit tests, which keeps the tests deterministic and avoids failures caused by temporary API availability.

## Knowledge Base Coverage

MetricMedic includes troubleshooting scenarios related to:

- duplicate records
- SQL joins
- NULL handling
- COUNT vs. COUNT DISTINCT
- date filters
- stale dashboards
- Power BI DAX context
- revenue reconciliation
- ETL failures
- incremental loads
- late-arriving data
- KPI definitions
- aggregation grain
- row-level security
- currency conversion
- historical data changes
- slowly changing dimensions
- fact and dimension table issues

New troubleshooting scenarios can be added directly to:

```text
data/troubleshooting_guide.json
```

## Security

Sensitive or local-only files are excluded using `.gitignore`, including:

```text
.env
metricmedic_chats.db
.venv/
__pycache__/
.pytest_cache/
```

The repository includes `.env.example` so other users know which environment variable is required without exposing the real Gemini API key.

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
