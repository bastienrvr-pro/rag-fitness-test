# RAG Fitness Chatbot 

**AI-powered fitness & nutrition advisor backed by scientific research**

A production-grade Retrieval-Augmented Generation (RAG) system specialized in fitness, nutrition, and strength training. Built with state-of-the-art semantic chunking, hybrid search, and cross-encoder reranking.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Recall@5](https://img.shields.io/badge/Recall@5-88.2%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📊 Performance Metrics

| Metric | Score | Benchmark |
|--------|-------|-----------|
| **Recall@5** | **88.2%** | Top 5-10% RAG systems |
| **MRR** | **0.776** | Correct doc at position 1.3 |
| **Precision@5** | **69.4%** | Excellent noise filtering |
| **Language** | **100% English** | Full consistency |

---

## ✨ Key Features

### 🧠 **State-of-Art Retrieval**
- **Semantic Chunking**: Similarity-based splitting (no mid-concept cuts)
- **Hybrid Search**: BM25 (sparse) + Dense (semantic) + Cross-Encoder reranking
- **BGE-Large Embeddings**: 1024-dimensional SOTA embeddings
- **Smart Filtering**: Document type and year filtering

### 💬 **User Experience**
- **Gradio Interface**: Clean, responsive chat UI
- **Thumbs Up/Down Feedback**: Track answer quality
- **Source Citations**: Transparent references with page numbers
- **Feedback Analytics**: Real-time statistics dashboard
- **Settings Panel**: Toggle hybrid search, adjust top-k

### 🔬 **Scientific Foundation**
- **4 Peer-Reviewed Papers**: ~1,000 semantic chunks
- **Topics Covered**: 
  - Protein requirements & supplementation
  - Training variables (volume, frequency, intensity)
  - Range of motion effects on hypertrophy
  - Bodybuilding nutrition strategies

### 🛡️ **Production-Ready**
- **Local Deployment**: 100% data sovereignty, 0€ running costs
- **Ollama Integration**: Local LLM (Llama 3.2 3B)
- **Feedback Export**: JSON format for continuous improvement
- **Optimized Prompts**: Low temperature (0.1) for factual answers

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  BM25 Search │  │ Dense Search │  │  RRF Fusion     │  │
│  │  (Keywords)  │  │  (Semantic)  │  │  (α = 0.5)      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                  │                    │           │
│         └──────────────────┴────────────────────┘           │
│                            │                                 │
│                            ▼                                 │
│                  ┌──────────────────┐                       │
│                  │  Cross-Encoder   │                       │
│                  │   Reranking      │                       │
│                  └────────┬─────────┘                       │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
                    Top-K Documents
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION PIPELINE                        │
│                                                              │
│  Context Augmentation + System Prompt                       │
│            │                                                 │
│            ▼                                                 │
│    Ollama (Llama 3.2 3B)                                    │
│            │                                                 │
│            ▼                                                 │
│  Factual Answer + Source Citations                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG-Fitness-Test/
├── notebooks/
│   ├── 01_build_knowledge_base.ipynb    # Semantic chunking + indexing
│   └── 02_evaluate_system.ipynb         # Metrics evaluation
│
├── src/
│   ├── config.py                        # System configuration
│   ├── retriever.py                     # Hybrid search implementation
│   └── chatbot.py                       # RAG pipeline
│
├── data/
│   ├── pdfs/                            # Scientific papers (4 PDFs)
│   │   ├── schoenfeld_rom_hypertrophy.pdf
│   │   ├── issn_protein_position.pdf
│   │   ├── helms_bodybuilding_nutrition.pdf
│   │   └── bernardez_training_variables.pdf
│   │
│   ├── golden_dataset.json              # 20 evaluation questions
│   ├── feedback.json                    # User feedback data
│   └── processed/
│       └── chroma_db/                   # Vector database (~1,000 chunks)
│
├── app.py                               # Gradio interface
├── requirements.txt                     # Dependencies
├── README.md                            # This file
└── EVALUATION_REPORT.md                 # Performance metrics
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- 8GB+ RAM recommended

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/rag-fitness-chatbot.git
cd rag-fitness-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama model
ollama pull llama3.2:3b

# 5. (Optional) Build knowledge base from scratch
# Open notebooks/01_build_knowledge_base.ipynb in Jupyter/VSCode
# Run all cells (~25 minutes)

# 6. Launch chatbot
python app.py
```

Open browser at: `http://127.0.0.1:7860`

---

## 💻 Usage

### Basic Questions

```
"What is the optimal protein intake for muscle hypertrophy?"
"Does full range of motion improve muscle growth?"
"How many sets per week for optimal volume?"
"Is creatine supplementation effective?"
```

### Advanced Queries

```
"Should protein intake be higher during a caloric deficit?"
"What is the mechanism behind full ROM producing more hypertrophy?"
"Is there a point where more volume stops helping muscle growth?"
```

### Providing Feedback

1. Ask a question
2. Read the answer
3. Click **👍 Helpful** or **👎 Not Helpful**
4. (Optional) Add a comment
5. Feedback saved to `data/feedback.json`

---

## 📊 Knowledge Base

### Scientific Papers

| Paper | Authors | Year | Topic | Chunks |
|-------|---------|------|-------|--------|
| ROM & Hypertrophy | Schoenfeld | 2016 | Range of motion effects | ~400 |
| Protein Position Stand | ISSN | 2017 | Protein requirements | ~100 |
| Bodybuilding Nutrition | Helms et al. | 2014 | Contest prep nutrition | ~150 |
| Training Variables | Bernárdez et al. | 2022 | Volume, frequency, intensity | ~150 |

**Total**: ~1,000 semantic chunks, 100% English

---

## ⚙️ Configuration

### Key Parameters (`src/config.py`)

```python
# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 1024 dimensions

# Generation
TEMPERATURE = 0.1        # Low = more factual
MAX_TOKENS = 256         # Concise answers

# Retrieval
TOP_K = 5                # Number of sources
HYBRID_RETRIEVE_K = 15   # Before reranking
HYBRID_ALPHA = 0.5       # BM25/Dense balance (0.5 = balanced)
```

### Hybrid Search Settings

- **α = 0.0**: BM25 only (keyword matching)
- **α = 0.5**: Balanced (recommended) ✅
- **α = 1.0**: Dense only (semantic)

---

## 🧪 Evaluation

### Run Evaluation Notebook

```bash
# Open notebooks/02_evaluate_system.ipynb
# Run all cells to get:
# - Recall@5
# - MRR (Mean Reciprocal Rank)
# - Precision@5
```

### Golden Dataset

20 curated questions across:
- **Nutrition** (7): Protein, creatine, BCAAs, timing
- **ROM** (5): Full vs partial range effects
- **Volume** (5): Sets, frequency, failure training
- **Out of scope** (3): Test refusal behavior

---

## 📈 Benchmarks

### Comparison with Baselines

| System | Chunking | Search | Recall@5 | MRR |
|--------|----------|--------|----------|-----|
| Baseline | Fixed-size | Dense only | 76.5% | 0.571 |
| + Hybrid Search | Fixed-size | BM25+Dense+Rerank | 82.4% | 0.623 |
| **Ultimate (This)** | **Semantic** | **BM25+Dense+Rerank** | **88.2%** | **0.776** |

**Improvement**: +11.7% Recall, +36% MRR 🚀

---

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.11**: Modern Python features
- **ChromaDB**: Vector database with HNSW index
- **Sentence-Transformers**: BGE-Large embeddings
- **Rank-BM25**: Sparse keyword search
- **Cross-Encoder**: Reranking (ms-marco-MiniLM)
- **Ollama**: Local LLM inference
- **Gradio**: Web interface

### Libraries

```
chromadb==0.4.22
sentence-transformers==2.2.2
rank-bm25==0.2.2
gradio==4.16.0
ollama-python==0.1.7
PyMuPDF==1.23.8
numpy==1.24.3
```

---

## 🗺️ Roadmap

### V2.0 (Future Enhancements)

- [ ] **Query Expansion**: Reformulate queries for better recall
- [ ] **Parent Document Retrieval**: Include surrounding context
- [ ] **Multilingual Support**: French, Spanish translations
- [ ] **More Papers**: Expand to 10-15 scientific papers
- [ ] **Fine-tuned Embeddings**: Domain-specific embeddings
- [ ] **API Deployment**: REST API with FastAPI
- [ ] **Monitoring Dashboard**: Metrics tracking (Prometheus/Grafana)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Additional Papers**: Submit PRs with new scientific papers
2. **Evaluation**: Add more questions to Golden Dataset
3. **UI/UX**: Improve Gradio interface
4. **Performance**: Optimize retrieval speed
5. **Documentation**: Improve guides and examples

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

### Scientific Papers

- Schoenfeld, B. (2016). Range of Motion Effects on Muscle Hypertrophy
- International Society of Sports Nutrition (2017). Protein Position Stand
- Helms, E. et al. (2014). Evidence-based Bodybuilding Contest Prep
- Bernárdez-Vázquez et al. (2022). Resistance Training Variables

### Open Source Projects

- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [Ollama](https://ollama.ai/)
- [Gradio](https://gradio.app/)

---

## 📧 Contact

**Project Maintainer**: [Your Name]
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Built with 💪 for the fitness community**
