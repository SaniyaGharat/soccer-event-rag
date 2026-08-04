"""
Match Event Retriever

Provides similarity search and metadata-filtered retrieval for soccer match events.
"""

from typing import List
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


class EventRetriever:
    """
    Retriever wrapper over Chroma vector store.
    """

    def __init__(self, vector_store: Chroma, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieves top-k relevant event documents for natural language query.
        """
        k = top_k or self.top_k
        docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def retrieve_with_scores(self, query: str, top_k: int = None):
        """
        Retrieves top-k relevant event documents along with similarity distance scores.
        """
        k = top_k or self.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)
