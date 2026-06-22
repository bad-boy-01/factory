import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class SmartChunker:
    """
    Smarter text chunking that respects chapter boundaries and paragraph structures.
    Ports logic from manhwa-video-factory Stage 02.
    """
    def __init__(self, target_words: int = 500, max_words: int = 800):
        self.target_words = target_words
        self.max_words = max_words

    def chunk_text(self, text: str) -> List[str]:
        # Split by any number of newlines to handle both single and double newline scripts
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        
        # If the entire novel has zero newlines, split by punctuation
        if len(paragraphs) == 1 and len(paragraphs[0].split()) > self.max_words:
            paragraphs = [p.strip() for p in re.split(r'(?<=[.!?。！？])\s+', text) if p.strip()]
            
        chunks = []
        current_chunk = []
        current_words = 0
        
        for p in paragraphs:
            p = p.strip()
            if not p: continue
            
            words = len(p.split())
            
            # Detect chapter boundary
            is_chapter = bool(re.match(r'^(chapter|ch\.|episode|volume|第.*章|제.*장)\s*\d+', p, re.IGNORECASE))
            
            # Condition 1: Chapter boundary and decent word count reached
            if is_chapter and current_words >= self.target_words * 0.5:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_words = words
                continue
                
            # Condition 2: Hard limit reached, force split
            if current_words + words > self.max_words:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [p]
                    current_words = words
                else:
                    # Single paragraph is larger than max_words (very rare)
                    current_chunk.append(p)
                    current_words += words
                continue
                
            current_chunk.append(p)
            current_words += words
            
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    def validate_chunks(self, original_text: str, chunks: List[str]) -> bool:
        orig_words = len(original_text.split())
        chunk_words = sum(len(c.split()) for c in chunks)
        
        # Text loss check (max 2% variance allowed for split artifacts)
        diff = abs(orig_words - chunk_words)
        margin = max(5, orig_words * 0.02)
        
        if diff > margin:
            logger.error(f"Chunk Validation Failed: Text loss detected! Original: {orig_words}, Chunks: {chunk_words}")
            return False
            
        logger.info(f"Chunk Validation Passed! (Original: {orig_words} words -> Chunks: {chunk_words} words)")
        return True
