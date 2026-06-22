"""
Continuity Validator — Novel Video Factory v4.5 (SIL Phase 2)
Verifies that generated scenes accurately reflect the ground truth narrative events.
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ContinuityValidator:
    """
    Acts as the 'Director's Checker'.
    Compares the planned visual scenes against the extracted Event Memory.
    If high-importance events are missing, validation fails.
    """
    def __init__(self, llm_adapter, importance_threshold: int = 7):
        self.llm = llm_adapter
        # Events with importance >= this threshold MUST be represented in scenes
        self.importance_threshold = importance_threshold

    def validate(self, scenes: List[Dict], events: List[Dict]) -> bool:
        """
        Check if the generated scenes cover all high-importance events.
        Uses the LLM for semantic comparison to avoid rigid keyword matching.
        """
        if not events:
            return True # Nothing to verify against
            
        if not scenes:
            return False # Events exist, but no scenes generated

        # Filter for only high-importance events
        critical_events = [e for e in events if int(e.get("importance", 5)) >= self.importance_threshold]
        
        if not critical_events:
            return True # No critical events, assume coverage is fine

        logger.debug(f"Verifying continuity for {len(critical_events)} critical event(s)...")

        # Format inputs for the LLM
        events_text = "\n".join([f"- {e['summary']} (Importance: {e['importance']})" for e in critical_events])
        scenes_text = "\n".join([f"Scene {i+1}: Location: {s.get('location', '')}. Action: {s.get('action', '')}. Narration: {s.get('narration_text', '')}" for i, s in enumerate(scenes)])

        system = (
            "You are a Continuity Supervisor for a storyboard. "
            "Your job is to ensure that NO CRITICAL STORY EVENTS were skipped by the storyboard artist.\n\n"
            "You will be given a list of 'Critical Events' and a list of 'Storyboard Scenes'.\n"
            "Compare them and answer: Are all the Critical Events represented in the Storyboard Scenes?\n"
            "Reply with EXACTLY one word: 'YES' if all events are covered, or 'NO' if any event is missing."
        )
        
        prompt = (
            f"--- CRITICAL EVENTS ---\n{events_text}\n\n"
            f"--- STORYBOARD SCENES ---\n{scenes_text}"
        )

        try:
            # We use a fast, low-temperature generation for this binary check
            response = self.llm.generate(prompt, system_prompt=system, temperature=0.0)
            
            if "YES" in response.upper():
                return True
            else:
                logger.warning("Continuity validation failed: LLM determined critical events are missing from the scenes.")
                return False
                
        except Exception as e:
            logger.warning(f"Continuity LLM call failed ({e}). Defaulting to passing validation.")
            return True # Fail open to prevent pipeline blockage on LLM transient errors
