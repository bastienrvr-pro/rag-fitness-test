"""
Configuration - RAG Fitness Chatbot
"""

from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "processed" / "chroma_db"

# ==============================================================================
# MODELS
# ==============================================================================

# Embedding model (for retrieval)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# LLM (Ollama)
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ==============================================================================
# CHROMADB
# ==============================================================================

COLLECTION_NAME = "fitness_knowledge_base"

# ==============================================================================
# RETRIEVAL PARAMETERS
# ==============================================================================

TOP_K = 5  # Number of documents to retrieve

# Hybrid Search parameters
HYBRID_RETRIEVE_K = 15  # Retrieve 15 before re-ranking (reduced from 20)
HYBRID_ALPHA = 0.5  # 0.5 = balanced BM25/Dense, 0=BM25 only, 1=Dense only

# ==============================================================================
# GENERATION PARAMETERS
# ==============================================================================

TEMPERATURE = 0.1  # Low temperature = more factual, less creative
MAX_TOKENS = 256   # Concise answers (reduced from 512)

# ==============================================================================
# SYSTEM PROMPT (English, Strict Rules)
# ==============================================================================

SYSTEM_PROMPT = """You are a fitness and nutrition expert assistant.

⚠️⚠️⚠️ CRITICAL RULES - MUST FOLLOW ⚠️⚠️⚠️

1. Answer ONLY using information from CONTEXT below
2. NEVER invent, assume, or extrapolate beyond the context
3. If information is missing → respond: "I don't have this information in my available sources"
4. ALWAYS cite sources: [Source: filename.pdf, page X]
5. Keep answers concise (maximum 200 words)
6. Answer in ENGLISH only

IMPORTANT NOTES:
- Focus on scientific evidence
- Cite specific studies and authors when available
- Be precise with numbers and recommendations
- If sources conflict, mention both perspectives

My knowledge covers:
- Sports nutrition (protein intake, supplements, meal timing)
- Training variables (volume, frequency, intensity, range of motion)
- Muscle hypertrophy mechanisms
- Evidence-based training recommendations

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (in English, factual, max 200 words, with source citations):"""
