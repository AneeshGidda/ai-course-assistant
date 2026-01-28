# Course Pilot - LLM Professor

An AI-powered educational platform that acts as a virtual professor, capable of:
- Answering questions about course materials
- Teaching concepts interactively
- Generating practice exam questions
- Creating and grading quizzes

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key (for embeddings)

### Database Setup

1. Start PostgreSQL with pgvector:
   ```bash
   docker-compose up -d
   ```

2. Initialize the database:
   ```bash
   python scripts/init_db.py
   ```

### Environment Variables

Create a `.env` file in the **project root** (same directory as `docker-compose.yml`):
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/coursepilot
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Optional, defaults to text-embedding-3-small
```

**Note:** The config will also check `backend/.env` as a fallback if the root `.env` doesn't exist.

### Ingestion

Ingest course materials:
```bash
python scripts/ingest_course.py CS479
```

This will:
- Parse all course files (PDF, PPTX, DOCX, MD, TXT)
- Chunk documents semantically
- Generate embeddings
- Store chunks in the vector database
- Prevent duplicates on re-ingestion

WORK IN PROGRESS (check issues and PRs to see progress)
