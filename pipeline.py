"""
Pipeline Orchestrator
Runs the complete RAG pipeline: parse → chunk → embed → chatbot
"""

from steps import parse, chunk, embed, chatbot


def run_full_pipeline():
    """Execute the complete RAG pipeline."""
    print("\n" + "🚀 " * 40)
    print("RAG PIPELINE")
    print("🚀 " * 40 + "\n")
    
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"
    CHUNKS_FOLDER = "chunks"
    DB_PATH = "chroma_db"
    COLLECTION_NAME = "rag_documents"
    WHISPER_MODEL = "base"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "google/flan-t5-xl"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    
    print("Pipeline steps:")
    print("  1. Parse documents (PDFs + audio/video)")
    print("  2. Chunk text into semantic segments")
    print("  3. Generate embeddings and store in database")
    print("  4. Launch interactive chatbot\n")
    
    response = input("Run full pipeline? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("STEP 1/4: PARSING")
    print("=" * 80 + "\n")
    
    pdf_files = parse.parse_pdfs(DATA_FOLDER, OUTPUT_FOLDER)
    audio_files = parse.parse_audio(DATA_FOLDER, OUTPUT_FOLDER, WHISPER_MODEL)
    
    total_parsed = len(pdf_files) + len(audio_files)
    
    if total_parsed == 0:
        print("\n✗ No files parsed. Please add PDFs or audio files to 'data/' folder.")
        return
    
    print("\n" + "=" * 80)
    print("STEP 2/4: CHUNKING")
    print("=" * 80 + "\n")
    
    chunk_files = chunk.chunk_documents(OUTPUT_FOLDER, CHUNKS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if not chunk_files:
        print("\n✗ No chunks created.")
        return
    
    print("\n" + "=" * 80)
    print("STEP 3/4: EMBEDDING")
    print("=" * 80 + "\n")
    
    chunks = embed.load_chunks(CHUNKS_FOLDER)
    embed.embed_and_store(chunks, DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL)
    
    print("\n" + "=" * 80)
    print("STEP 4/4: CHATBOT")
    print("=" * 80 + "\n")
    
    bot = chatbot.RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    bot.chat()


if __name__ == "__main__":
    run_full_pipeline()
