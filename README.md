# RAG Pipeline

A clean, production-ready Retrieval-Augmented Generation system for processing educational content (PDFs and audio/video files).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python pipeline.py

# Or run steps individually
python steps/parse.py
python steps/chunk.py
python steps/embed.py
python steps/chatbot.py
```

## Features

- **Document Processing**: PDFs (LangChain PyPDFLoader) and audio/video (Whisper)
- **Semantic Chunking**: Content-aware splitting with LangChain RecursiveCharacterTextSplitter
- **State-of-the-art Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions)
- **Powerful LLM**: Google Flan-T5-XL (3B parameters, 2048 token context)
- **Vector Database**: ChromaDB with persistent storage
- **Clean Architecture**: Modular design with separate scripts for each step

## Architecture

```
data/                    # Input: PDFs and audio/video files
├── lecture1.pdf
├── meeting1.mp4
└── ...

output/                  # Parsed text files
├── lecture1.txt
├── meeting1.txt
└── ...

chunks/                  # Chunked documents
├── lecture1_chunks.txt
├── meeting1_chunks.txt
└── ...

chroma_db/              # Vector database (persistent)

steps/                  # Pipeline steps
├── parse.py            # Step 1: Document parsing
├── chunk.py            # Step 2: Semantic chunking
├── embed.py            # Step 3: Generate embeddings
└── chatbot.py          # Step 4: Interactive Q&A

pipeline.py             # Orchestrates all steps
requirements.txt        # Dependencies
```

## Pipeline Steps

### 1. Parse (steps/parse.py)
- Extracts text from PDFs using LangChain's PyPDFLoader
- Transcribes audio/video files using Whisper (base model)
- Output: `.txt` files in `output/` folder

### 2. Chunk (steps/chunk.py)
- Detects content type (presentation vs transcription)
- Applies content-aware separators for optimal chunking
- Uses RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
- Output: `_chunks.txt` files in `chunks/` folder

### 3. Embed (steps/embed.py)
- Generates embeddings with BAAI/bge-base-en-v1.5
- Stores in ChromaDB with metadata (source, chunk_id)
- Output: Persistent vector database in `chroma_db/`

### 4. Chatbot (steps/chatbot.py)
- Retrieves top 5 relevant chunks per query
- Generates answers with Flan-T5-XL
- Interactive Q&A loop

## Configuration

All settings are defined as constants in each file:

**steps/parse.py**
- `DATA_FOLDER = "data"`
- `OUTPUT_FOLDER = "output"`
- `WHISPER_MODEL = "base"`

**steps/chunk.py**
- `CHUNK_SIZE = 1000`
- `CHUNK_OVERLAP = 150`

**steps/embed.py**
- `EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"`
- `COLLECTION_NAME = "rag_documents"`

**steps/chatbot.py**
- `LLM_MODEL = "google/flan-t5-xl"`
- `n_results = 5` (retrieved chunks)

## Testing

1. Place sample files in `data/` folder:
   - PDFs (lecture slides, presentations)
   - MP4/MP3/WAV files (meeting recordings, lectures)

2. Run pipeline:
   ```bash
   python pipeline.py
   ```

3. Ask questions:
   - "What are the best practices for production RAG?"
   - "What databases are recommended for GenAI?"
   - "Explain the RAG architecture patterns"

## Requirements

- Python 3.11 (recommended)
- ~10GB disk space for models
- 8GB+ RAM recommended for Flan-T5-XL

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Transcription | Whisper Base | ~140MB | Audio → Text |
| Embeddings | BAAI/bge-base-en-v1.5 | ~440MB | Text → Vectors |
| Generation | Flan-T5-XL | ~11GB | Context + Query → Answer |

## Performance

- **Parsing**: ~2-5 min/hour of audio (Whisper base)
- **Chunking**: < 1 second for typical documents
- **Embedding**: ~10-50 chunks/second
- **Retrieval**: < 100ms per query
- **Generation**: ~1-3 seconds per answer (CPU)

## License

MIT
