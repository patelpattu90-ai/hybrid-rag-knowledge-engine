"""
compressor.py — Contextual Compression

Extracts relevant sentences from retrieved chunks using LLM-as-compressor.
If compression fails or drops everything, always falls back to originals.

FIXED: Previous version dropped all chunks when LLM returned IRRELEVANT
for every chunk on broad queries. Now uses a two-tier fallback:
  1. If a chunk is marked IRRELEVANT → keep original (not drop)
  2. Only drop if compressed version is shorter than MIN_COMPRESSED_CHARS
     AND the original chunk is also very short
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

_MAX_CHUNKS_TO_COMPRESS = 5
_MIN_COMPRESSED_CHARS   = 40

_COMPRESS_SYSTEM = """You are a document compression assistant.

Given a QUERY and a CONTEXT CHUNK, extract ONLY the sentences from the chunk
that are directly relevant to answering the query.

Rules:
- Copy relevant sentences VERBATIM — do not rephrase or summarise
- If multiple sentences are relevant, include all of them
- If NO sentences are relevant, output exactly: IRRELEVANT
- Do not add any preamble, explanation, or formatting
- Output only the extracted sentences (or IRRELEVANT)"""


@dataclass
class CompressionResult:
    original_chunks:   list[str]
    compressed_chunks: list[str]
    dropped:           int
    original_chars:    int
    compressed_chars:  int

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 1.0
        return round(self.compressed_chars / self.original_chars, 3)

    def badge(self) -> str:
        pct = int((1 - self.compression_ratio) * 100)
        return (
            f"🗜️ **Compression:** {len(self.original_chunks)} → "
            f"{len(self.compressed_chunks)} chunks  ·  "
            f"{pct}% tokens saved  ·  "
            f"{self.dropped} irrelevant chunk(s) dropped"
        )


class ContextualCompressor:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"

    def _compress_one(self, query: str, chunk: str) -> str:
        """
        Compress a single chunk.
        ALWAYS returns a non-empty string — falls back to original on any failure
        or IRRELEVANT response. Never returns None or empty string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM},
                    {"role": "user",   "content": f"QUERY: {query}\n\nCONTEXT CHUNK:\n{chunk}"},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            extracted = response.choices[0].message.content.strip()

            # LLM said nothing relevant — keep original so we never lose context
            if not extracted or extracted.upper() == "IRRELEVANT":
                logger.info("Chunk marked IRRELEVANT — keeping original as fallback")
                return chunk

            # LLM returned something longer than original — keep original
            if len(extracted) > len(chunk) * 1.1:
                return chunk

            # Compressed version is too short to be useful — keep original
            if len(extracted) < _MIN_COMPRESSED_CHARS:
                return chunk

            return extracted

        except Exception as e:
            logger.warning(f"Compression call failed: {e} — keeping original")
            return chunk   # always safe fallback

    def compress(self, query: str, contexts: list[str]) -> CompressionResult:
        """
        Compress up to _MAX_CHUNKS_TO_COMPRESS chunks.
        Returns CompressionResult — compressed_chunks always has content.
        """
        if not contexts:
            return CompressionResult(
                original_chunks=[], compressed_chunks=[],
                dropped=0, original_chars=0, compressed_chars=0,
            )

        original_chars = sum(len(c) for c in contexts)
        compressed     = []
        dropped        = 0

        for i, chunk in enumerate(contexts):
            if i < _MAX_CHUNKS_TO_COMPRESS:
                result = self._compress_one(query, chunk)
                compressed.append(result)
                if result == chunk:
                    # count as "not compressed" but NOT dropped
                    pass
            else:
                # Beyond cap — include uncompressed
                compressed.append(chunk)

        # Final safety net — should never trigger now but kept for safety
        if not compressed:
            logger.warning("compressed list empty — using originals")
            compressed = list(contexts)

        compressed_chars = sum(len(c) for c in compressed)

        return CompressionResult(
            original_chunks=contexts,
            compressed_chunks=compressed,
            dropped=dropped,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
        )
