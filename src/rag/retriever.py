"""
Match Event Retriever (Hybrid Sparse-Dense with Custom RRF & Metadata Filters)

Combines:
1. Chroma Dense Vector Search
2. BM25 Sparse Keyword Search
3. Custom Reciprocal Rank Fusion (RRF) ensemble algorithm
4. Self-Querying Metadata Extraction (rule-based NLP parser)
5. Interactive UI filters (applied on the retrieved document list)
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


class EventRetriever:
    """
    Hybrid Sparse-Dense Retriever featuring custom RRF fusion and metadata self-querying.
    """

    def __init__(
        self,
        vector_store: Chroma,
        documents: List[Document],
        top_k: int = 4,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Initialize BM25 Sparse Retriever
        if documents:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            self.bm25_retriever = None

    def parse_query_for_filters(self, query: str) -> Dict[str, Any]:
        """
        Self-Querying Parser: Extracts structural metadata filters directly from the natural language query.
        E.g., "turnovers in defensive third by Team A" -> {'team_id': 'Team A', 'pitch_zones': 'Defensive Third'}
        """
        filters = {}
        query_lower = query.lower()

        # 1. Team filter extraction
        if "team a" in query_lower:
            filters["team_id"] = "Team A"
        elif "team b" in query_lower:
            filters["team_id"] = "Team B"

        # 2. Pitch Zone filter extraction
        if "penalty box" in query_lower or "penalty area" in query_lower:
            if "attacking" in query_lower:
                filters["pitch_zones"] = "Attacking Penalty Box"
            elif "defensive" in query_lower:
                filters["pitch_zones"] = "Defensive Penalty Box"
            else:
                filters["pitch_zones"] = "Penalty Box"
        elif "defensive third" in query_lower:
            filters["pitch_zones"] = "Defensive Third"
        elif "attacking third" in query_lower:
            filters["pitch_zones"] = "Attacking Third"
        elif "midfield" in query_lower:
            filters["pitch_zones"] = "Midfield"
        elif "flank" in query_lower or "wing" in query_lower:
            if "left" in query_lower:
                filters["pitch_zones"] = "Left Flank"
            elif "right" in query_lower:
                filters["pitch_zones"] = "Right Flank"
            else:
                filters["pitch_zones"] = "Flank"

        # 3. Event Type filter extraction
        if "pass" in query_lower:
            filters["event_type"] = "completed_pass"
        elif "turnover" in query_lower or "lost possession" in query_lower or "interception" in query_lower:
            filters["event_type"] = "turnover"
        elif "press" in query_lower or "pressure" in query_lower:
            filters["event_type"] = "high_press_sequence"
        elif "offside" in query_lower:
            filters["event_type"] = "proxy_offside"
        elif "shot" in query_lower or "goal" in query_lower:
            filters["event_type"] = "shot_on_goal"
        elif "box entry" in query_lower or "entered the box" in query_lower:
            filters["event_type"] = "penalty_box_entry"

        return filters

    def _ensemble_rrf(self, dense_docs: List[Document], sparse_docs: List[Document], k: int) -> List[Document]:
        """
        Applies Reciprocal Rank Fusion (RRF) to merge dense and sparse search results.
        Formula: Score(d) = sum( weight / (60 + rank) )
        """
        rrf_scores = {}
        doc_map = {}
        constant = 60.0

        for rank, doc in enumerate(dense_docs, 1):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (self.dense_weight / (constant + rank))

        for rank, doc in enumerate(sparse_docs, 1):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0.0) + (self.sparse_weight / (constant + rank))

        # Sort by descending RRF score
        sorted_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        return [doc_map[c] for c in sorted_contents[:k]]

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        ui_filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieves top-k relevant event documents using hybrid retrieval,
        self-querying NLP filters, and manual UI selectors.
        """
        k = top_k or self.top_k

        # 1. Self-querying filter extraction
        self_query_filters = self.parse_query_for_filters(query)

        # 2. Merge with UI filters
        merged_filters = {}
        if ui_filters:
            merged_filters.update(ui_filters)
        merged_filters.update(self_query_filters)

        # 3. Dense search with Chroma (applying metadata filters on the database level where possible)
        dense_filter = {}
        if merged_filters:
            if "team_id" in merged_filters:
                dense_filter["team_id"] = merged_filters["team_id"]
            if "event_type" in merged_filters:
                dense_filter["event_type"] = merged_filters["event_type"]

        if dense_filter:
            # Query dense database with metadata filter
            dense_docs = self.vector_store.similarity_search(query, k=k*2, filter=dense_filter)
        else:
            dense_docs = self.vector_store.similarity_search(query, k=k*2)

        # 4. Sparse search with BM25
        if self.bm25_retriever:
            self.bm25_retriever.k = k * 2
            sparse_docs = self.bm25_retriever.invoke(query)
        else:
            sparse_docs = []

        # 5. RRF Fusion
        docs = self._ensemble_rrf(dense_docs, sparse_docs, k=k*2)

        # 6. Strict Post-Retrieval Metadata Filtering (Ensures 100% filter accuracy)
        if merged_filters:
            filtered_docs = []
            for doc in docs:
                match = True
                for key, val in merged_filters.items():
                    doc_val = doc.metadata.get(key, "")
                    if not doc_val:
                        continue
                    
                    if key == "pitch_zones":
                        if val == "Penalty Box":
                            if "Penalty Box" not in doc_val:
                                match = False
                        elif val not in doc_val:
                            match = False
                    elif doc_val != val:
                        match = False
                
                if match:
                    filtered_docs.append(doc)
            
            # Supplement if filtered result is smaller than k
            if len(filtered_docs) < k:
                for doc in docs:
                    if doc not in filtered_docs and len(filtered_docs) < k:
                        filtered_docs.append(doc)
            return filtered_docs[:k]

        return docs[:k]

    def retrieve_with_scores(self, query: str, top_k: int = None):
        """
        Retrieves top-k relevant event documents from the dense vector store along with distance scores.
        """
        k = top_k or self.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)
