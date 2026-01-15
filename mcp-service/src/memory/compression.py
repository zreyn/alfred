"""
Semantic compression using LLM to extract atomic facts from dialogues.
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

import httpx

from memory.config import OLLAMA_HOST, LLM_MODEL, LLM_TEMPERATURE
from memory.models import Dialogue, AtomicFact

logger = logging.getLogger(__name__)

# Prompt template for extracting atomic facts
COMPRESSION_PROMPT = """You are a memory compression system. Your task is to extract atomic facts from a conversation.

An atomic fact is:
- A self-contained statement that makes sense without context
- Has all pronouns resolved to actual names (e.g., "he" -> "John")
- Has relative times converted to absolute (e.g., "tomorrow" -> the actual date)
- Contains one piece of information

For each fact, extract:
- content: The complete, unambiguous restatement
- keywords: Key terms for search
- persons: Names of people mentioned
- locations: Places mentioned
- entities: Other notable entities (organizations, products, etc.)
- timestamp: When the fact occurred (ISO format if determinable, null otherwise)
- topic: Brief topic category

Return a JSON array of facts. If no meaningful facts can be extracted, return an empty array.

Example output:
[
  {
    "content": "John Smith's birthday is on March 15th, 1990.",
    "keywords": ["birthday", "john smith", "march"],
    "persons": ["John Smith"],
    "locations": [],
    "entities": [],
    "timestamp": "1990-03-15",
    "topic": "personal information"
  }
]

CONVERSATION:
{dialogue_text}

CURRENT DATE: {current_date}

Extract atomic facts as JSON:"""


class SemanticCompressor:
    """Compresses dialogues into atomic facts using LLM."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        timeout: float = 60.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _format_dialogues(self, dialogues: list[Dialogue]) -> str:
        """Format dialogues into text for the LLM."""
        lines = []
        for d in dialogues:
            timestamp_str = d.timestamp.isoformat() if d.timestamp else ""
            if timestamp_str:
                lines.append(f"[{timestamp_str}] {d.speaker}: {d.content}")
            else:
                lines.append(f"{d.speaker}: {d.content}")
        return "\n".join(lines)

    def _parse_facts(self, response_text: str) -> list[AtomicFact]:
        """Parse LLM response into AtomicFact objects."""
        # Try to extract JSON from response
        try:
            # First try direct JSON parse
            facts_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            match = re.search(r'\[[\s\S]*\]', response_text)
            if match:
                try:
                    facts_data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON from LLM response")
                    return []
            else:
                logger.warning("No JSON array found in LLM response")
                return []

        if not isinstance(facts_data, list):
            logger.warning("LLM response is not a list")
            return []

        facts = []
        for item in facts_data:
            if not isinstance(item, dict) or "content" not in item:
                continue

            timestamp = None
            if item.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(item["timestamp"])
                except (ValueError, TypeError):
                    pass

            fact = AtomicFact(
                content=item["content"],
                keywords=item.get("keywords", []),
                persons=item.get("persons", []),
                locations=item.get("locations", []),
                entities=item.get("entities", []),
                timestamp=timestamp,
                topic=item.get("topic"),
            )
            facts.append(fact)

        return facts

    async def compress_dialogues(
        self,
        dialogues: list[Dialogue],
    ) -> list[AtomicFact]:
        """Compress dialogues into atomic facts.

        Args:
            dialogues: List of Dialogue objects to compress.

        Returns:
            List of AtomicFact objects extracted from the dialogues.
        """
        if not dialogues:
            return []

        client = await self._get_client()
        url = f"{self.host}/api/generate"

        # Format the prompt
        dialogue_text = self._format_dialogues(dialogues)
        current_date = datetime.now().strftime("%Y-%m-%d")

        prompt = COMPRESSION_PROMPT.format(
            dialogue_text=dialogue_text,
            current_date=current_date,
        )

        try:
            response = await client.post(
                url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            response_text = data.get("response", "")
            logger.debug(f"LLM response: {response_text[:500]}...")

            facts = self._parse_facts(response_text)
            logger.info(f"Extracted {len(facts)} atomic facts from {len(dialogues)} dialogues")

            return facts

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama generate API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Failed to compress dialogues: {e}")
            raise
