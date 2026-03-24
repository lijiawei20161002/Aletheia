"""
Cross-document campaign detection using ChromaDB + sentence-transformers.

Detects coordinated narrative campaigns by finding clusters of semantically
similar propaganda documents. When multiple documents share the same core
framing and rhetorical structure within a time window, it suggests coordinated
production rather than independent reporting.

Falls back gracefully to in-memory cosine similarity if ChromaDB is not
installed — useful in development or CI environments.
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

_SIMILARITY_THRESHOLD = 0.72   # cosine similarity above which → coordinated narrative
_MIN_DOCS_FOR_CAMPAIGN = 2     # at least N matching docs to flag a campaign


@dataclass
class CampaignMatch:
    doc_id: str
    similarity: float
    propaganda_score: int
    snippet: str
    source_url: Optional[str] = None


@dataclass
class CampaignResult:
    is_campaign: bool
    matches: List[CampaignMatch]
    campaign_score: float   # 0–1, highest similarity found

    def summary(self) -> str:
        if not self.is_campaign:
            return (
                f"No coordinated campaign detected "
                f"(highest similarity: {self.campaign_score:.2f})"
            )
        return (
            f"CAMPAIGN DETECTED — {len(self.matches)} similar document(s) found "
            f"(confidence: {self.campaign_score:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "is_campaign": self.is_campaign,
            "campaign_score": self.campaign_score,
            "matches": [
                {
                    "doc_id": m.doc_id,
                    "similarity": m.similarity,
                    "propaganda_score": m.propaganda_score,
                    "snippet": m.snippet,
                    "source_url": m.source_url,
                }
                for m in self.matches
            ],
        }


class CampaignDetector:
    """
    Stores analyzed documents and queries for near-duplicate narratives.

    Usage
    -----
    detector = CampaignDetector()

    # After each /analyze call:
    detector.add_document(text, propaganda_score=8, source_url="https://...")
    result = detector.find_similar(text)
    if result.is_campaign:
        print(result.summary())
    """

    def __init__(self, persist_path: str = "/tmp/aletheia_campaign_db"):
        self._persist_path = persist_path
        self._collection = None
        self._model = None
        self._fallback_store: List[Dict[str, Any]] = []
        self._use_chromadb = self._try_init_chromadb()

    def _try_init_chromadb(self) -> bool:
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path=self._persist_path)
            self._collection = chroma_client.get_or_create_collection(
                name="propaganda_docs",
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, text: str) -> List[float]:
        self._load_model()
        return self._model.encode(text, show_progress_bar=False).tolist()

    @staticmethod
    def _doc_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(
        self,
        text: str,
        propaganda_score: int,
        source_url: Optional[str] = None,
    ) -> str:
        """
        Store a document for future campaign detection.

        Returns the document's stable ID (SHA-256 prefix of text).
        Silently ignores duplicate documents.
        """
        doc_id = self._doc_id(text)
        snippet = text[:200].replace("\n", " ")
        metadata = {
            "propaganda_score": propaganda_score,
            "source_url": source_url or "",
            "timestamp": time.time(),
            "snippet": snippet,
        }

        try:
            embedding = self._embed(text)
        except ImportError:
            embedding = None

        if self._use_chromadb and embedding is not None:
            try:
                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata],
                )
            except Exception:
                pass  # Duplicate IDs → already stored, which is fine
        else:
            # Fallback: check for duplicate before appending
            if not any(e["id"] == doc_id for e in self._fallback_store):
                self._fallback_store.append({
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                })

        return doc_id

    def find_similar(
        self,
        text: str,
        n_results: int = 5,
    ) -> CampaignResult:
        """
        Query for documents with similar narrative to `text`.

        Excludes the document itself from results.
        """
        doc_id = self._doc_id(text)

        if self._use_chromadb:
            return self._query_chromadb(text, doc_id, n_results)
        else:
            return self._query_fallback(text, doc_id, n_results)

    def collection_size(self) -> int:
        if self._use_chromadb and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return 0
        return len(self._fallback_store)

    # ------------------------------------------------------------------
    # Internal query backends
    # ------------------------------------------------------------------

    def _query_chromadb(self, text: str, doc_id: str, n_results: int) -> CampaignResult:
        try:
            embedding = self._embed(text)
            # Request one extra to account for self-match exclusion
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results + 1, max(self._collection.count(), 1)),
                include=["distances", "metadatas", "documents"],
            )
        except Exception:
            return CampaignResult(is_campaign=False, matches=[], campaign_score=0.0)

        matches = []
        for rid, dist, meta, doc in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            if rid == doc_id:
                continue
            # ChromaDB cosine distance → similarity
            sim = max(0.0, 1.0 - dist)
            if sim >= _SIMILARITY_THRESHOLD:
                matches.append(CampaignMatch(
                    doc_id=rid,
                    similarity=round(sim, 3),
                    propaganda_score=int(meta.get("propaganda_score", 0)),
                    snippet=meta.get("snippet", doc[:200]),
                    source_url=meta.get("source_url") or None,
                ))

        campaign_score = max((m.similarity for m in matches), default=0.0)
        return CampaignResult(
            is_campaign=len(matches) >= _MIN_DOCS_FOR_CAMPAIGN,
            matches=sorted(matches, key=lambda m: m.similarity, reverse=True),
            campaign_score=round(campaign_score, 3),
        )

    def _query_fallback(self, text: str, doc_id: str, n_results: int) -> CampaignResult:
        try:
            import torch
            from sentence_transformers import util
            query_emb = self._embed(text)
        except (ImportError, Exception):
            return CampaignResult(is_campaign=False, matches=[], campaign_score=0.0)

        q = torch.tensor(query_emb).unsqueeze(0)
        matches = []
        for entry in self._fallback_store:
            if entry["id"] == doc_id or entry["embedding"] is None:
                continue
            e = torch.tensor(entry["embedding"]).unsqueeze(0)
            sim = float(util.cos_sim(q, e)[0][0])
            if sim >= _SIMILARITY_THRESHOLD:
                meta = entry["metadata"]
                matches.append(CampaignMatch(
                    doc_id=entry["id"],
                    similarity=round(sim, 3),
                    propaganda_score=int(meta.get("propaganda_score", 0)),
                    snippet=meta.get("snippet", entry["text"][:200]),
                    source_url=meta.get("source_url") or None,
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        matches = matches[:n_results]
        campaign_score = max((m.similarity for m in matches), default=0.0)
        return CampaignResult(
            is_campaign=len(matches) >= _MIN_DOCS_FOR_CAMPAIGN,
            matches=matches,
            campaign_score=round(campaign_score, 3),
        )
