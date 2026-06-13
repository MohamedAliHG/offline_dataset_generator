"""ChromaDB store wrapper used across ingestion and generation."""

import logging
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Minimal wrapper around ChromaDB with an app-specific interface."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "collection_demo",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        drop_old: bool = False,
        namespace: str = "default",
        device: str = "cpu",
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        del uri, db_name

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.namespace = namespace

        logger.info("Loading embeddings model %s on %s", embed_model, device)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        if drop_old:
            try:
                self.client.delete_collection(collection_name)
                logger.info("Dropped existing collection: %s", collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore ready collection='%s' persist_dir='%s' docs=%s",
            collection_name,
            persist_directory,
            self.collection.count(),
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            logger.warning("add_documents called with empty list")
            return []

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)

            ids = [
                f"{self.collection_name}_{i}_{abs(hash(text)) % 10**10}"
                for i, text in enumerate(texts)
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info("Indexed %s documents", len(ids))
            return ids
        except Exception as exc:
            logger.error("Failed to index documents: %s", exc, exc_info=True)
            return []

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        namespace: Optional[str] = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)

        where = filter or {}
        ns = namespace or self.namespace
        if ns:
            ns_filter = {"namespace": {"$eq": ns}}
            where = {**where, **ns_filter} if where else ns_filter
        where = where or None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Document] = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def query_chunks(
        self,
        query: str,
        k: int = 4,
        namespace: Optional[str] = None,
        filter: Optional[dict] = None,
    ) -> List[Dict]:
        """Return scored chunks using the same embedding model used at index time."""
        query_embedding = self.embeddings.embed_query(query)

        where = filter or {}
        ns = namespace or self.namespace
        if ns:
            ns_filter = {"namespace": {"$eq": ns}}
            where = {**where, **ns_filter} if where else ns_filter
        where = where or None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[Dict] = []
        for chunk_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": meta or {},
                    "distance": distance,
                }
            )
        return chunks

    def delete_by_namespace(self, namespace: str) -> None:
        try:
            self.collection.delete(where={"namespace": {"$eq": namespace}})
            logger.info("Deleted docs for namespace='%s'", namespace)
        except Exception as exc:
            logger.error("Failed deleting namespace '%s': %s", namespace, exc, exc_info=True)

    def count(self) -> int:
        return self.collection.count()
