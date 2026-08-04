"""
Chroma Vector Store Manager

Initializes Chroma vector store and indexes adaptively-chunked match event documents.
"""

import os
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def get_embedding_function(model_name: str = "all-MiniLM-L6-v2"):
    """Returns local HuggingFace embedding model."""
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"[!] Warning: Failed to load HuggingFace model '{model_name}': {e}")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class VectorStoreManager:
    """
    Manages indexing and persistence for Chroma Vector DB.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "soccer_match_events",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_directory
        self.collection_name = collection_name
        self.embeddings = get_embedding_function(embedding_model_name)
        self.vector_store: Optional[Chroma] = None

    def index_documents(self, documents: List[Document], reset: bool = True) -> Chroma:
        """
        Indexes a list of event Documents into the Chroma vector store.
        """
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        if reset and os.path.exists(self.persist_dir):
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir
            )
        else:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            if documents:
                self.vector_store.add_documents(documents)

        return self.vector_store

    def load_vector_store(self) -> Chroma:
        """
        Loads existing persisted Chroma vector store.
        """
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        return self.vector_store
