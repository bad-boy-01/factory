"""
Scene Planner — Novel Video Factory v4
Converts text chunks into structured narrative scenes and cinematic shots.
"""
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class ScenePlanner:
    """
    Breaks text chunks into narrative scenes (the 'showrunner' role).
    Each scene groups related sentences with location/character/mood info.
    """
    def __init__(self, llm_adapter, config: dict = None):
        self.llm = llm_adapter
        self.config = config or {}

    def plan_scenes(self, text_chunk: str, chapter: int = 1, events: List[Dict] = None) -> List[Dict]:
        """Convert a text chunk into a list of visual scenes, using events as a skeleton."""
        system = (
            "You are a cinematic storyboard director.\n"
            "Turn the provided source_text into a JSON array of sequential scenes.\n\n"
            "RULES:\n"
            "1. NO SUMMARIZATION: Every sentence must be its own scene. One sentence = one scene.\n"
            "2. 100% COVERAGE: The 'narration_text' of all scenes combined must equal the source text 100%.\n"
            "3. USE EVENTS: Use the provided events as mandatory coverage targets. Every event must appear in at least one scene.\n"
            "4. USE SOURCE TEXT: Use source_text for visual details, dialogue context, emotion, atmosphere, clothing, and scene composition.\n"
            "5. CAMERA MOTION: Pick the single best fit from this exact list for "
            "'camera_motion': zoom_in, zoom_out, pan_left, pan_right, static. "
            "zoom_in for tense/intimate/shocking beats, zoom_out for "
            "isolation/sadness, pan_left/pan_right for wide establishing or "
            "traveling shots, static for calm dialogue with little motion.\n"
            "6. OUTPUT ONLY VALID JSON: Return a JSON object with a 'scenes' array.\n\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "scenes": [\n'
            "    {\n"
            '      "scene_id": "SC001",\n'
            '      "location": "Room",\n'
            '      "characters": ["Xu"],\n'
            '      "emotion": "neutral",\n'
            '      "action": "Xu sits.",\n'
            '      "camera_angle": "medium shot",\n'
            '      "camera_motion": "static",\n'
            '      "lighting": "daylight",\n'
            '      "visual_prompt_tags": "1boy",\n'
            '      "narration_text": "Xu sits.",\n'
            '      "complexity": 5\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        payload = {
            "events": events or [],
            "source_text": text_chunk[:3000]
        }
        prompt = json.dumps(payload, indent=2)

        max_t = self.config.get("models", {}).get("llm", {}).get("scene_max_tokens", 2500)
        response = self.llm.generate_json(prompt, system_prompt=system, temperature=0.2, max_tokens=max_t)
        
        try:
            scenes = json.loads(response)
            if isinstance(scenes, dict) and "scenes" in scenes:
                scenes = scenes["scenes"]
            if not isinstance(scenes, list):
                if isinstance(scenes, dict):
                    scenes = [scenes]
                else:
                    scenes = []
        except Exception as e:
            logger.warning(f"Scene planner JSON parse failed: {e}")
            scenes = []

        if not scenes:
            logger.warning("Scene planner returned empty or parse failed. Returning empty list to trigger retry.")
            return []

        _VALID_MOTIONS = {"zoom_in", "zoom_out", "pan_left", "pan_right", "static"}

        # Ensure all required fields are present
        for sc in scenes:
            sc.setdefault("location", "Unknown")
            sc.setdefault("characters", [])
            sc.setdefault("emotion", "neutral")
            sc.setdefault("action", "continuation")
            sc.setdefault("camera_angle", "medium shot")
            sc.setdefault("lighting", "cinematic lighting")
            sc.setdefault("visual_prompt_tags", "")
            sc.setdefault("narration_text", "")
            sc.setdefault("complexity", 5)
            # If the LLM omits camera_motion or returns something off-list,
            # leave it unset rather than guessing here — the renderer's
            # existing emotion/camera_angle heuristic is the fallback for
            # any scene that doesn't have a valid explicit motion.
            motion = str(sc.get("camera_motion", "")).strip().lower()
            sc["camera_motion"] = motion if motion in _VALID_MOTIONS else None

        return scenes

