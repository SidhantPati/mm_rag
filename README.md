# RAG Chat Application

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, and Google Gemini.

## Features

- **RAG Pipeline**: Retrieves relevant document chunks to augment LLM responses
- **ChromaDB**: Vector database for storing and searching document embeddings
- **Google Generative AI**: Uses Gemini for both text embeddings and chat responses
- **Chat History**: Maintains conversation history for context-aware responses
- **Environment Management**: Secure API key management using `python-dotenv`

## Prerequisites

- Python 3.9+
- Google Gemini API Key

## Usage

The main entry point is `answer_generation.py`. You can run it directly:

```bash
/Users/sidrocks/.gemini/antigravity-ide/scratch/mm_rag/.venv/bin/python /Users/sidrocks/.gemini/antigravity-ide/scratch/mm_rag/answer_generation.py
```

## Configuration

The following constants can be adjusted in `answer_generation.py`:

- `gemini_model`: The Gemini model to use for chat responses (default: `"gemini-3.5-flash-lite"`)
- `text_embedding_model`: The HuggingFace model for embeddings (default: `"BAAI/bge-large-en-v1.5"`)
- `vector_db_name`: Directory for ChromaDB persistence (default: `"chroma_vector_db"`)

## Project Structure

```
mm_rag/
├── .env                    # Environment variables (not in git)
├── .venv/                  # Python virtual environment
├── answer_generation.py    # Main RAG application
├── vector_ingestion.py     # Document Ingestion into ChromaDB Vector db
├── knowledge-base/         # Document source files
│   ├── company
│       ├── about.md
│       ├── careers.md
│       ...
│   ├── contracts
│   ├── employees
│   └── products
└── README.md               # Project documentation
```

## Development

### Embedding Documents

To embed documents and populate the vector database, run:

```bash
/Users/sidrocks/.gemini/antigravity-ide/scratch/mm_rag/.venv/bin/python /Users/sidrocks/.gemini/antigravity-ide/scratch/mm_rag/embedding.py
```

This will:
1.  Read `.md` files from the `knowledge-base/*` directory
2.  Split them into manageable chunks
3.  Generate embeddings using `BAAI/bge-large-en-v1.5`
4.  Store them in `chroma_vector_db/`

## Troubleshooting

- **API Key Issues**: Ensure `GOOGLE_API_KEY` is set in your `.env` file and that you have access to the specified Gemini model.
- **ChromaDB Not Found**: Run `vector_ingestion.py` first to populate the vector database.
- **Module Not Found**: Ensure you've activated the virtual environment: `source .venv/bin/activate`.
