"""
Hybrid retrieval combining semantic and keyword search.
"""

import logging
import re
from typing import Optional

from memory.config import SEMANTIC_TOP_K, KEYWORD_TOP_K
from memory.embeddings import OllamaEmbeddings
from memory.vectordb import MemoryVectorDB
from memory.models import MemorySearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines semantic and keyword search for memory retrieval."""

    def __init__(
        self,
        vector_db: MemoryVectorDB,
        embeddings: OllamaEmbeddings,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from a query string."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "about", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "i", "me", "my",
            "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves",
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        semantic_limit: int = SEMANTIC_TOP_K,
        keyword_limit: int = KEYWORD_TOP_K,
    ) -> list[MemorySearchResult]:
        """Retrieve memories using hybrid search.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            semantic_limit: Maximum results from semantic search.
            keyword_limit: Maximum results from keyword search.

        Returns:
            List of MemorySearchResult sorted by combined score.
        """
        # Run semantic search
        query_embedding = await self.embeddings.embed(query)
        semantic_results = await self.vector_db.search_semantic(
            query_embedding,
            limit=semantic_limit,
        )

        # Run keyword search
        keywords = self._extract_keywords(query)
        keyword_results = []
        if keywords:
            keyword_results = await self.vector_db.search_keyword(
                keywords,
                limit=keyword_limit,
            )

        # Merge and deduplicate results
        seen_ids = set()
        combined_results: dict[str, MemorySearchResult] = {}

        # Add semantic results with weighted scores
        for result in semantic_results:
            memory_id = result.memory.id
            if memory_id not in combined_results:
                combined_results[memory_id] = MemorySearchResult(
                    memory=result.memory,
                    score=result.score * self.semantic_weight,
                )
            else:
                combined_results[memory_id].score += result.score * self.semantic_weight

        # Add keyword results with weighted scores
        for result in keyword_results:
            memory_id = result.memory.id
            if memory_id not in combined_results:
                combined_results[memory_id] = MemorySearchResult(
                    memory=result.memory,
                    score=result.score * self.keyword_weight,
                )
            else:
                combined_results[memory_id].score += result.score * self.keyword_weight

        # Sort by combined score and limit
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:limit]

    async def retrieve_semantic_only(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """Retrieve memories using semantic search only.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of MemorySearchResult sorted by semantic similarity.
        """
        query_embedding = await self.embeddings.embed(query)
        return await self.vector_db.search_semantic(query_embedding, limit=limit)
