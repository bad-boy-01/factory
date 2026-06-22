"""
Translation Pipeline — Novel Video Factory v4

Smart language detection:
1. Fast heuristic: if >85% ASCII characters → likely English → SKIP
2. LLM confirmation if heuristic is uncertain (saves tokens)
3. Full translation only when genuinely needed

Supports: Korean → English, Chinese → English, any language → English
If source is already English: returns text unchanged (zero LLM calls).
"""
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


def _is_primarily_ascii(text: str, threshold: float = 0.85) -> bool:
    """Fast check: if >85% chars are ASCII, text is likely English."""
    if not text:
        return True
    sample = text[:2000]  # Only check first 2000 chars for speed
    ascii_chars = sum(1 for c in sample if ord(c) < 128)
    return (ascii_chars / len(sample)) >= threshold


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (handles English, Chinese, Korean punctuation)."""
    pattern = r"(?<=[.!?。！？\n])\s+"
    parts = re.split(pattern, text.strip())
    return [p.strip() for p in parts if p.strip()]


class TranslationPipeline:
    """
    Layer 2: Translation with smart language detection.
    Skips translation entirely if text is already English.
    """
    def __init__(self, config: dict, llm_adapter):
        self.config = config or {}
        self.llm = llm_adapter
        self.source_lang = (self.config.get("project", {})
                            .get("language", {}).get("source", "auto"))
        self.target_lang = (self.config.get("project", {})
                            .get("language", {}).get("target", "English"))

    def is_translation_needed(self, text: str) -> bool:
        """
        Returns True if text needs to be translated.
        Uses fast heuristic first, then LLM as fallback.
        """
        # If source is explicitly set to English, skip immediately
        if self.source_lang.lower() == "english":
            return False

        # Fast heuristic: ASCII ratio
        if _is_primarily_ascii(text):
            logger.info("Language check: ASCII ratio high → text is English, skipping translation")
            return False

        # Slow path: ask LLM to confirm (only for ambiguous cases)
        if self.llm is not None:
            system = (
                "Detect the language of the following text. "
                "Reply with EXACTLY one word: 'ENGLISH' if it is entirely English, "
                "or 'OTHER' if it contains any non-English language."
            )
            try:
                response = self.llm.generate(text[:500], system_prompt=system, temperature=0.0)
                if "ENGLISH" in response.upper():
                    logger.info("LLM confirmed: text is English, skipping translation")
                    return False
            except Exception as e:
                logger.warning(f"Language detection LLM call failed: {e}")

        logger.info(f"Translation needed: source={self.source_lang} → {self.target_lang}")
        return True

    def process_chapter(self, text: str) -> str:
        """Translate if needed, otherwise return as-is."""
        if not self.is_translation_needed(text):
            return text

        max_chars = 2000  # Translate in ~2000-char chunks to avoid token limits
        chunks = self._split_for_translation(text, max_chars=max_chars)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            logger.info(f"  Translating chunk {i+1}/{len(chunks)}…")
            result = self._translate_chunk(chunk)
            translated_parts.append(result or chunk)

        return "\n\n".join(translated_parts)

    def _translate_chunk(self, text: str) -> str:
        system = (
            f"You are a professional literary translator. "
            f"Translate the following text to {self.target_lang}. "
            "RULES: "
            "1. Translate every sentence — never skip or summarize. "
            "2. Preserve all character names, dialogue formatting, and scene directions. "
            "3. Preserve the author's tone and style. "
            "Output ONLY the translated text, nothing else."
        )
        if self.llm is None:
            return text  # No LLM: return original
        return self.llm.generate(text, system_prompt=system, temperature=0.3) or text

    def _split_for_translation(self, text: str, max_chars: int = 2000) -> List[str]:
        """Split text into chunks that don't break mid-sentence."""
        sentences = _split_sentences(text)
        chunks, current, length = [], [], 0
        for s in sentences:
            if length + len(s) > max_chars and current:
                chunks.append(" ".join(current))
                current, length = [], 0
            current.append(s)
            length += len(s)
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]
