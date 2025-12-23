"""
Retriever - Hybrid Search for RAG Fitness

Features:
- Dense search (BGE-large embeddings)
- Hybrid search (BM25 + Dense + Cross-Encoder reranking)
- Optimized for semantic chunks
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import config

class Retriever:
    """
    Retriever with Hybrid Search capabilities
    
    Methods:
        - search(): Dense search (cosine similarity)
        - hybrid_search(): BM25 + Dense + Cross-Encoder reranking
    """
    
    def __init__(self):
        """Initialize retriever with ChromaDB and hybrid search components"""
        
        print("🔧 Initializing Retriever...")
        
        # ====================================================================
        # EMBEDDINGS (Dense Search)
        # ====================================================================
        
        print(f"   📥 Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # ====================================================================
        # CHROMADB
        # ====================================================================
        
        print(f"   💾 Connecting to ChromaDB: {config.CHROMA_DIR}")
        
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_collection(config.COLLECTION_NAME)
        
        doc_count = self.collection.count()
        print(f"   ✅ Collection '{config.COLLECTION_NAME}': {doc_count} documents")
        
        # ====================================================================
        # BM25 (Sparse Search)
        # ====================================================================
        
        print("   🔤 Initializing BM25 index...")
        
        # Retrieve all documents for BM25
        all_data = self.collection.get(include=["documents"])
        
        self.all_documents = all_data['documents']
        self.all_ids = all_data['ids']
        
        # Tokenize corpus (simple whitespace tokenization)
        self.tokenized_corpus = [doc.lower().split() for doc in self.all_documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"      ✅ BM25 indexed: {len(self.all_documents)} documents")
        
        # ====================================================================
        # CROSS-ENCODER (Re-ranking)
        # ====================================================================
        
        print("   🎯 Loading Cross-Encoder for re-ranking...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("      ✅ Cross-Encoder loaded")
        
        print("\n✅ Retriever ready (Hybrid Search enabled)")
    
    
    def search(
        self,
        query: str,
        top_k: int = None,
        doc_type: Optional[str] = "scientific_paper",
        min_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Dense search (cosine similarity on embeddings)
        
        Args:
            query: User question
            top_k: Number of results
            doc_type: Filter by document type
            min_year: Filter documents >= year
        
        Returns:
            List of documents with metadata and scores
        """
        
        if top_k is None:
            top_k = config.TOP_K
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        # Build filter
        where_filter = {}
        if doc_type:
            where_filter["type"] = doc_type
        
        # Retrieve more if post-filtering needed
        retrieve_k = top_k * 3 if min_year else top_k
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Post-filter and format
        documents = []
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Year filter
            if min_year:
                doc_year = meta.get('year')
                if doc_year and int(doc_year) < min_year:
                    continue
            
            documents.append({
                'text': doc,
                'source': meta['source'],
                'authors': meta.get('authors', 'Unknown'),
                'year': meta.get('year', 'Unknown'),
                'journal': meta.get('journal', 'Unknown'),
                'page': meta.get('page', 'N/A'),
                'section': meta.get('section', 'body'),
                'score': 1 - dist
            })
            
            if len(documents) >= top_k:
                break
        
        return documents
    
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        retrieve_k: int = None,
        alpha: float = None,
        doc_type: Optional[str] = "scientific_paper",
        min_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Hybrid search: BM25 + Dense + Cross-Encoder reranking
        
        Args:
            query: User question
            top_k: Final number of results
            retrieve_k: Number before reranking
            alpha: Fusion weight (0=BM25, 1=Dense, 0.5=balanced)
            doc_type: Filter by document type
            min_year: Filter documents >= year
        
        Returns:
            List of reranked documents
        """
        
        if top_k is None:
            top_k = config.TOP_K
        if retrieve_k is None:
            retrieve_k = config.HYBRID_RETRIEVE_K
        if alpha is None:
            alpha = config.HYBRID_ALPHA
        
        # ====================================================================
        # STEP 1: BM25 SEARCH
        # ====================================================================
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:retrieve_k]
        
        # ====================================================================
        # STEP 2: DENSE SEARCH
        # ====================================================================
        
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        where_filter = {}
        if doc_type:
            where_filter["type"] = doc_type
        
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # ====================================================================
        # STEP 3: RRF FUSION
        # ====================================================================
        
        fusion_scores = {}
        k_rrf = 60
        
        for rank, idx in enumerate(bm25_top_indices, 1):
            doc_id = self.all_ids[idx]
            rrf_score = 1 / (k_rrf + rank)
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + alpha * rrf_score
        
        for rank, doc_id in enumerate(dense_results['ids'][0], 1):
            rrf_score = 1 / (k_rrf + rank)
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + (1 - alpha) * rrf_score
        
        fused_top_ids = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:retrieve_k]
        
        # ====================================================================
        # STEP 4: CROSS-ENCODER RE-RANKING
        # ====================================================================
        
        fused_ids = [doc_id for doc_id, _ in fused_top_ids]
        fused_docs_data = self.collection.get(
            ids=fused_ids,
            include=["documents", "metadatas"]
        )
        
        pairs = [[query, doc] for doc in fused_docs_data['documents']]
        rerank_scores = self.reranker.predict(pairs)
        reranked_indices = np.argsort(rerank_scores)[::-1]
        
        # ====================================================================
        # STEP 5: FORMAT
        # ====================================================================
        
        documents = []
        
        for idx in reranked_indices:
            meta = fused_docs_data['metadatas'][idx]
            
            if min_year:
                doc_year = meta.get('year')
                if doc_year and int(doc_year) < min_year:
                    continue
            
            documents.append({
                'text': fused_docs_data['documents'][idx],
                'source': meta['source'],
                'authors': meta.get('authors', 'Unknown'),
                'year': meta.get('year', 'Unknown'),
                'journal': meta.get('journal', 'Unknown'),
                'page': meta.get('page', 'N/A'),
                'section': meta.get('section', 'body'),
                'score': float(rerank_scores[idx])
            })
            
            if len(documents) >= top_k:
                break
        
        return documents
    
    
    def format_context(self, documents: List[Dict]) -> str:
        """Format documents for LLM prompt"""
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"""
Document {i}:
Source: {doc['source']}
Authors: {doc['authors']}
Year: {doc['year']}
Page: {doc['page']}

{doc['text']}
""")
        
        return "\n".join(context_parts)
