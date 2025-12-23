# 📊 RAG FITNESS - EVALUATION REPORT

Date: 2025-12-23 18:37

---

## 🎯 GOLDEN DATASET

- **Total questions**: 20
- **In-scope questions**: 17
- **Categories**:
  - nutrition: 7 questions
  - rom: 5 questions
  - volume: 5 questions
  - out_of_scope: 3 questions

---

## 🔍 RETRIEVER METRICS

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Recall@5** | 88.2% | ✅ Excellent |
| **MRR** | 0.776 | Average position: 1.3 |
| **Precision@5** | 69.4% | 69% of retrieved docs are relevant |

**Evaluation**: 17 in-scope questions

**Interpretation**:
- Recall > 80%: ✅ Excellent
- Recall 60-80%: ⚠️ Acceptable
- Recall < 60%: ❌ Needs improvement

---

## 🤖 GENERATOR QUALITY

**Evaluation method**: Manual inspection of 5 sample answers

**Note**: LLM-as-Judge with small models (Llama 3.2 3B) is not reliable.
Manual inspection is recommended for assessing:
- Faithfulness (no hallucinations)
- Completeness (answers the question fully)
- Relevance (concise and useful)

**To evaluate generator quality**:
1. Run this notebook
2. Review sample answers in Step 5
3. Rate each answer manually (1-5 scale)
4. Average scores give true quality estimate

---

## 📁 FILES

- Golden Dataset: `data/golden_dataset.json`
- Knowledge Base: `data/processed/chroma_db/`
- Evaluation Notebook: `notebooks/02_evaluate_system.ipynb`
