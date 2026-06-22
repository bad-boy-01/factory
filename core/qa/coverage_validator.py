import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class CoverageValidator:
    """
    Ensures LLM didn't skip story chunks by comparing word counts.
    Ports logic from manhwa-video-factory Stage 07c.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def validate(self, source_text: str, scenes: List[Dict]) -> bool:
        if not source_text or not source_text.strip():
            return True
            
        source_words = len(source_text.split())
        narration_words = sum(len(str(sc.get("narration_text", "")).split()) for sc in scenes)
        
        if source_words == 0:
            return True
            
        ratio = narration_words / source_words
        logger.info(f"Coverage check: {narration_words}/{source_words} words ({ratio:.1%})")
        
        if ratio < self.threshold:
            logger.warning(f"Low coverage detected: {ratio:.1%} (target >={self.threshold:.1%})")
            return False
            
        return True
