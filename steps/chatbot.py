"""
RAG Chatbot
Interactive chatbot with retrieval-augmented generation
"""

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
import torch


class RAGChatbot:
    def __init__(self, 
                 db_path: str = "chroma_db",
                 collection_name: str = "rag_documents",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "google/flan-t5-xl"):
        """Initialize RAG chatbot with retrieval and generation models."""
        print("=" * 80)
        print("INITIALIZING RAG CHATBOT")
        print("=" * 80)
        
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {llm_model}\n")
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded\n")
        
        print("Loading LLM (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float32
        )
        print("✓ LLM loaded\n")
        
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(name=collection_name)
        print(f"✓ Connected to collection: {collection_name}")
        print(f"  Total chunks in database: {self.collection.count()}\n")
        
        print("=" * 80)
        print("CHATBOT READY")
        print("=" * 80)
    
    def retrieve(self, query: str, n_results: int = 5) -> Tuple[List[str], List[dict]]:
        """Retrieve relevant chunks from vector database."""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        return chunks, metadatas
    
    def generate_answer(self, query: str, context: str, max_length: int = 2048) -> str:
        """Generate answer using LLM based on retrieved context."""
        prompt = f"""Answer the question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        outputs = self.llm.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def chat(self):
        """Interactive chat loop."""
        print("\nRAG Chatbot (type 'quit' or 'exit' to stop)\n")
        
        while True:
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print()
            
            chunks, metadatas = self.retrieve(query)
            
            sources = list(set([meta['source'] for meta in metadatas]))
            print(f"Retrieved {len(chunks)} relevant chunks from: {', '.join(sources)}\n")
            
            context = "\n\n".join(chunks)
            answer = self.generate_answer(query, context)
            
            print(f"Answer: {answer}\n")
            print("-" * 80 + "\n")


if __name__ == "__main__":
    DB_PATH = "chroma_db"
    COLLECTION_NAME = "rag_documents"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "google/flan-t5-xl"
    
    chatbot = RAGChatbot(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL)
    chatbot.chat()
