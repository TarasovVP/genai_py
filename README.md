# Data Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Vertex AI](https://img.shields.io/badge/Vertex_AI-Gemini-4285F4.svg?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An AI-powered data assistant that generates relational synthetic datasets from SQL DDL, loads them into PostgreSQL, and answers natural-language questions with guarded NL2SQL and visualizations.

The project combines three workflows in one Streamlit application:

1. **DDL → synthetic data** — parse a relational schema and generate realistic, constraint-aware table data with Gemini on Vertex AI.
2. **Natural language → data patch** — update, delete, or add generated rows through structured AI operations.
3. **Question → SQL → insight** — turn a data question into read-only PostgreSQL, execute it, and display a table or chart.

## Features

- Parses SQL DDL into an internal schema model.
- Normalizes common MySQL syntax for PostgreSQL, including `AUTO_INCREMENT`, `DATETIME`, and `ENUM`.
- Understands primary keys, foreign keys, unique constraints, `NOT NULL`, and `CHECK ... IN (...)` values.
- Generates tables in dependency order so child rows can reuse keys from their parent tables.
- Uses Gemini structured output with JSON Schema, retry, repair, and token-expansion logic.
- Splits large generations into adaptive batches.
- Corrects generated primary keys, foreign keys, and constrained values after model output.
- Saves every dataset as DDL, schema metadata, and CSV files.
- Exports one table as CSV or the complete dataset as ZIP.
- Applies AI-generated patch operations instead of regenerating an entire table.
- Converts natural-language questions into guarded, read-only PostgreSQL queries.
- Renders tabular results, bar charts, line charts, and histograms.
- Masks common PII patterns and blocks basic prompt-injection and off-topic requests.
- Traces AI generations, database operations, guardrails, and charts through optional Langfuse integration.

## How it works

```text
SQL DDL
  │
  ▼
DDL normalization and parsing
  │
  ▼
Table dependency ordering
  │
  ▼
Gemini structured data generation
  │
  ▼
PK / FK / allowed-value normalization
  │
  ├──► CSV and ZIP export
  ├──► AI patch editing
  └──► PostgreSQL
          │
          ▼
      Natural-language question
          │
          ▼
      Guardrails → NL2SQL → read-only check
          │
          ▼
      Result table and visualization
```

## Tech stack

- **UI:** Streamlit
- **AI:** Gemini through Google Vertex AI and the `google-genai` SDK
- **Data processing:** Pandas
- **Database:** PostgreSQL 16 and Psycopg 3
- **Observability:** Langfuse
- **Local infrastructure:** Docker Compose

The exact Python dependency versions are pinned in [`requirements.txt`](requirements.txt).

## Project structure

```text
genai_py/
├── app.py                         # Streamlit entry point and navigation
├── config.py                      # Environment-based configuration
├── state.py                       # Streamlit session state
├── ddl_parser.py                  # DDL parser and schema extraction
├── data_generator.py              # Dependency-aware synthetic data generation
├── data_editor.py                 # Structured AI patch generation and application
├── vertex_client.py               # Vertex AI structured output, retries, and repair
├── postgres_client.py             # PostgreSQL DDL, insert, and connection operations
├── screens/
│   ├── data_generation_page.py    # Generation, preview, export, and editing UI
│   └── talk_to_data_page.py       # Guarded NL2SQL chat UI
├── domain/
│   ├── ddl_normalizer.py          # MySQL-like DDL normalization
│   ├── nl2sql.py                  # NL2SQL prompts and execution flow
│   ├── guardrails.py              # PII, injection, and topic checks
│   ├── sql_guard.py               # Read-only SQL validation
│   ├── allowed_values.py          # CHECK / ENUM value normalization
│   ├── fk_allowed_values.py       # Foreign-key value resolution
│   ├── schema_render.py           # Compact schema representation for prompts
│   └── charts.py                  # Result visualization
├── services/                      # Traced Vertex AI and PostgreSQL wrappers
├── storage/                       # Dataset persistence and exports
├── tracing/                       # Langfuse integration
└── docker-compose.yml             # Local PostgreSQL 16
```

## Requirements

- Python 3.11+
- Docker with Docker Compose
- A Google Cloud project with Vertex AI enabled
- Application Default Credentials with permission to use Vertex AI

Langfuse is optional. The application works without Langfuse credentials.

## Quick start

### 1. Clone the repository

```shell
git clone https://github.com/TarasovVP/genai_py.git
cd genai_py
```

### 2. Create a virtual environment

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

### 3. Authenticate with Google Cloud

```shell
gcloud auth application-default login
```

Make sure the selected Google Cloud project has Vertex AI enabled and your account can invoke the configured model.

### 4. Configure the environment

`VERTEX_PROJECT` is required:

```shell
export VERTEX_PROJECT="your-gcp-project"
export VERTEX_LOCATION="europe-west1"
export VERTEX_MODEL="gemini-2.0-flash-001"
```

The repository ignores `.env`, but the application does not load it automatically. If you keep the values in `.env`, export them before starting the app:

```shell
set -a
source .env
set +a
```

### 5. Start PostgreSQL

```shell
docker compose up -d
```

### 6. Run the application

```shell
streamlit run app.py
```

Open `http://localhost:8501`.

## Configuration

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `VERTEX_PROJECT` | Yes | — | Google Cloud project with Vertex AI enabled |
| `VERTEX_LOCATION` | No | `europe-west1` | Vertex AI region |
| `VERTEX_MODEL` | No | `gemini-2.0-flash-001` | Gemini model deployed in the selected region |
| `DEFAULT_ROWS_PER_TABLE` | No | `10` | Number of rows generated for each table |
| `DEFAULT_SEED` | No | `0` | Python random seed; `0` leaves it unset |
| `DATASETS_ROOT` | No | `datasets` | Directory used for generated datasets |
| `PG_HOST` | No | `localhost` | PostgreSQL host |
| `PG_PORT` | No | `55432` | PostgreSQL port exposed by Docker Compose |
| `PG_DB` | No | `data_assistant` | Database name |
| `PG_USER` | No | `data_assistant` | Database user |
| `PG_PASSWORD` | No | `data_assistant` | Database password for local development |
| `LANGFUSE_PUBLIC_KEY` | No | empty | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | empty | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse host |

## Example DDL

Save the following schema as `demo.sql` and upload it on the **Data Generation** page:

```sql
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    department_id INTEGER NOT NULL REFERENCES departments(id),
    name VARCHAR(120) NOT NULL,
    email VARCHAR(200) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'inactive')),
    salary NUMERIC(10, 2),
    hired_at DATE
);
```

Optionally add a global instruction such as:

```text
Generate a realistic software company dataset for Germany. Use EUR salaries and diverse employee names.
```

The application generates parent tables first, passes their identifiers to dependent tables, saves the result, and loads it into PostgreSQL.

## AI-assisted editing

Select a generated table and describe the desired change:

```text
Activate all inactive employees and add three senior engineers.
```

Gemini returns structured operations using only:

- `update_where`
- `delete_where`
- `add_rows`

The application applies the patch locally, validates types and constraints, updates the saved CSV, and reloads the table in PostgreSQL.

## Talk to your data

Open **Talk to your data** after generating a dataset and ask questions such as:

```text
Show the five departments with the highest average salary as a bar chart.
```

Gemini returns a structured response containing:

- one PostgreSQL query;
- a short explanation;
- the expected result kind;
- an optional chart specification.

Before execution, the application rejects empty, multi-statement, non-`SELECT`, and known data-modifying SQL.

## Dataset storage

Each generation is stored locally under a unique identifier:

```text
datasets/<dataset-id>/
├── ddl.sql
├── schema.json
├── meta.json
├── departments.csv
└── employees.csv
```

The `datasets/` directory is excluded from version control.

## Observability

When Langfuse credentials are configured, the application records:

- generation phases and model parameters;
- prompts and structured outputs;
- PostgreSQL reload and query operations;
- guardrail decisions;
- execution duration and errors;
- chart rendering metadata.

Review your data-handling requirements before enabling remote tracing: prompts, SQL, output previews, and metadata may be sent to the configured Langfuse instance.

## Security and operational notes

- The NL2SQL path masks common email, phone, card, and SSN patterns before prompting the model.
- Basic prompt-injection and off-topic requests are rejected.
- Generated SQL must pass an application-level read-only check.
- These checks are defense in depth, not a substitute for database permissions. Use a dedicated PostgreSQL role with only the required access outside local development.
- Every full dataset load runs `DROP SCHEMA IF EXISTS public CASCADE` on the configured database. Use the included dedicated local PostgreSQL container, never a shared or production database.
- The Docker Compose credentials are development defaults and must be replaced for any exposed environment.
- Generated data and model output should be reviewed before downstream use.

## Current scope

- Generated datasets are saved on disk, but the UI does not yet reopen a previous dataset after an application restart.
- Chat results remain visible during the Streamlit session; follow-up questions are currently generated from the latest question and schema rather than the full conversation history.
- The custom DDL parser targets common relational schemas. Vendor-specific SQL, cyclic dependencies, and composite foreign keys may require manual adjustment.
- Histogram rendering requires Matplotlib. Install it separately with `pip install matplotlib` if needed.

## License

Licensed under the [Apache License 2.0](LICENSE).
