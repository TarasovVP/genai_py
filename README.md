# Data Assistant

An intelligent data assistant built with Streamlit and Google Vertex AI. The application allows generating synthetic data from DDL schemas and executing natural language queries to databases.

## Features

### 🔄 Data Generation
- Upload DDL schemas to create table structures
- Automatic synthetic data generation using Google Gemini 2.0 Flash
- Support for various data types (INTEGER, VARCHAR, DATE, BOOLEAN, etc.)
- Edit generated data through AI assistant
- Save datasets in CSV format

### 💬 Natural Language Queries
- Natural language queries to data (NL2SQL)
- Automatic SQL query generation using Vertex AI
- Result visualization (charts, histograms)
- Secure query execution with SQL injection protection
- Query and result history

## Tech Stack

### Frontend
- **Streamlit** - web application interface
- **Pandas** - data processing and analysis

### Backend & AI
- **Google Vertex AI** - data and SQL query generation
- **Gemini 2.0 Flash** - main language model
- **Langfuse** - AI operations tracing and monitoring

### Database
- **PostgreSQL 16** - main DBMS
- **Docker Compose** - containerization

### Key Libraries
- `streamlit` - web interface
- `pandas` - data manipulation
- `google-genai` - Vertex AI client
- `psycopg[binary]` - PostgreSQL driver

## Project Structure

```
genai_py/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and settings
├── data_generator.py         # Synthetic data generation
├── ddl_parser.py            # DDL schema parser
├── vertex_client.py         # Vertex AI client
├── postgres_client.py       # PostgreSQL client
├── data_editor.py           # Data editing
├── state.py                 # Session state management
├── screens/                 # Application pages
│   ├── data_generation_page.py
│   └── talk_to_data_page.py
├── services/                # Services with logging
│   ├── vertex_logged.py
│   └── postgres_logged.py
├── domain/                  # Business logic
│   ├── nl2sql.py           # Natural language to SQL
│   ├── sql_guard.py        # SQL injection protection
│   ├── charts.py           # Visualization
│   └── guardrails.py       # Protection mechanisms
├── storage/                # Storage operations
├── tracing/                # Operations tracing
├── datasets/               # Generated datasets
└── docker-compose.yml     # Docker configuration
```

## Installation and Setup

### 1. Clone repository
```bash
git clone <repository-url>
cd genai_py
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # for Linux/Mac
# or
.venv\Scripts\activate     # for Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create `.env` file with the following parameters:
```env
# Vertex AI
VERTEX_PROJECT=your-gcp-project
VERTEX_LOCATION=europe-west1
VERTEX_MODEL=gemini-2.0-flash-001

# PostgreSQL
PG_HOST=localhost
PG_PORT=55432
PG_DB=data_assistant
PG_USER=data_assistant
PG_PASSWORD=data_assistant

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Data generation settings
DEFAULT_ROWS_PER_TABLE=10
DEFAULT_SEED=0
DATASETS_ROOT=datasets
```

### 5. Start PostgreSQL
```bash
docker-compose up -d
```

### 6. Run application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

### Data Generation
1. Navigate to "Data Generation" page
2. Enter DDL table schemas in the text field
3. Configure generation parameters (row count, seed)
4. Click "Generate Data" to create synthetic data
5. Edit data if needed
6. Save the dataset

### Data Queries
1. Navigate to "Talk to your data" page
2. Select a saved dataset
3. Enter a natural language query
4. Get results as table or chart
5. View the generated SQL query

## Features

### Security
- All SQL queries are safety-checked before execution
- SQL injection and data modification protection
- Read-only database operations only

### AI Capabilities
- Intelligent realistic data generation
- Automatic data type detection
- Contextual natural language understanding
- Adaptive result visualization

### Monitoring
- AI operations tracing via Langfuse
- Query and response logging
- Performance metrics

## Requirements

- Python 3.11+
- Docker and Docker Compose
- Google Cloud account with Vertex AI access
- PostgreSQL (via Docker)
