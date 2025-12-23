"""
RAG Chatbot - Complete pipeline with Hybrid Search
"""

import requests
from typing import Dict, Optional

from retriever import Retriever
import config


class RAGChatbot:
    """Complete RAG pipeline with Hybrid Search"""
    
    def __init__(self):
        """Initialize retriever and verify Ollama"""
        
        print("\n" + "="*80)
        print("🤖 INITIALIZING RAG CHATBOT")
        print("="*80)
        
        self.retriever = Retriever()
        
        print(f"\n🔍 Checking Ollama ({config.OLLAMA_BASE_URL})...")
        
        try:
            response = requests.get(config.OLLAMA_BASE_URL, timeout=5)
            
            if response.status_code == 200:
                print("   ✅ Ollama is available")
                print(f"   🧠 Model: {config.OLLAMA_MODEL}")
                print(f"   🌡️ Temperature: {config.TEMPERATURE}")
                print(f"   📏 Max tokens: {config.MAX_TOKENS}")
            else:
                print("   ⚠️ Ollama responding but unexpected status")
        
        except Exception as e:
            print(f"   ❌ Ollama not accessible: {e}")
            print("   → Run 'ollama serve' in another terminal")
        
        print("\n✅ Chatbot ready!")
        print("="*80 + "\n")
    
    
    def answer(
        self,
        question: str,
        doc_type: Optional[str] = "scientific_paper",
        min_year: Optional[int] = None,
        top_k: int = None,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Generate answer to a question
        
        Args:
            question: User question
            doc_type: Filter by document type
            min_year: Filter documents >= year
            top_k: Number of documents
            use_hybrid: Use Hybrid Search
        
        Returns:
            Dict with answer, sources, context, method
        """
        
        if top_k is None:
            top_k = config.TOP_K
        
        # ====================================================================
        # RETRIEVAL
        # ====================================================================
        
        if use_hybrid:
            documents = self.retriever.hybrid_search(
                query=question,
                top_k=top_k,
                retrieve_k=15,
                alpha=0.5,
                doc_type=doc_type,
                min_year=min_year
            )
        else:
            documents = self.retriever.search(
                query=question,
                top_k=top_k,
                doc_type=doc_type,
                min_year=min_year
            )
        
        # ====================================================================
        # AUGMENTATION
        # ====================================================================
        
        context = self.retriever.format_context(documents)
        
        # ====================================================================
        # GENERATION
        # ====================================================================
        
        prompt = config.SYSTEM_PROMPT.format(
            context=context,
            question=question
        )
        
        answer = self._call_ollama(prompt)
        
        # ====================================================================
        # RETURN
        # ====================================================================
        
        return {
            'answer': answer,
            'sources': documents,
            'context': context,
            'method': 'hybrid' if use_hybrid else 'dense'
        }
    
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama for generation"""
        
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "temperature": config.TEMPERATURE,
                    "stream": False,
                    "options": {
                        "num_predict": config.MAX_TOKENS
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Ollama error (status {response.status_code})"
        
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    
    def format_answer_with_sources(self, result: Dict) -> str:
        """Format answer with sources for display"""
        
        answer = result['answer']
        sources = result['sources']
        method = result.get('method', 'unknown')
        
        sources_text = "\n" + "─" * 80 + "\n"
        sources_text += "📚 SOURCES USED\n\n"
        
        for i, source in enumerate(sources, 1):
            sources_text += f"{i}. {source['source']}\n"
            sources_text += f"   * Authors: {source['authors']}\n"
            sources_text += f"   * Year: {source['year']}\n"
            sources_text += f"   * Page: {source['page']}\n"
            sources_text += f"   * Score: {source['score']:.3f}\n"
            
            if i == 1 and method == 'hybrid':
                sources_text += f"   * Method: Hybrid Search (BM25 + Dense + Reranking)\n"
            
            sources_text += "\n"
        
        return f"💬 ANSWER\n\n{answer}\n\n{sources_text}"
