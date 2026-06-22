"""
Storyboard Planner — Novel Video Factory v5
Extracts visual narrative beats from text chunks to generate a flat storyboard sequence.
Maintains continuity state across chunks.
"""
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class StoryboardPlanner:
    """
    Breaks narrative blocks into cinematic visual beats.
    Maintains environment state (location, characters, time, weather) across blocks.
    """
    def __init__(self, llm_adapter, config: dict = None):
        self.llm = llm_adapter
        self.config = config or {}

        self.VALID_BEATS = {
            "environment", "action", "reaction", "emotion",
            "dialogue", "object_focus", "reveal", "combat", "transition"
        }
        self.VALID_SHOTS = {
            "establishing_shot", "wide_shot", "medium_shot",
            "close_up", "extreme_close_up", "over_shoulder"
        }

    def plan_panels(self, text_chunk: str, current_state: dict, chapter: int = 1, start_sequence: int = 1) -> dict:
        """
        Convert a text chunk into a list of visual panels.
        Returns a dict containing 'state' and 'panels'.
        """
        system = (
            "You are a cinematic storyboard director for a manga/manhwa adaptation.\n"
            "Turn the provided narrative block into a JSON storyboard.\n\n"
            "RULES:\n"
            "1. EXTRACT VISUAL BEATS: Do NOT extract sentence-by-sentence. Combine related actions into a single beat (e.g. fishing, sitting, casting -> 1 action beat). Include internal thoughts as emotion beats.\n"
            "2. MAINTAIN CONTINUITY: Use the Provided State. Update the State based on the text.\n"
            "3. BEAT TYPES:\n"
            "   - 'environment': Establishing shots, scenery.\n"
            "   - 'emotion': Character thoughts, internal monologue, reactions.\n"
            "   - 'reveal': Sudden interruptions, massive new elements bursting in.\n"
            "   - 'object_focus': INANIMATE objects only (magic rings, weapons, floats). Animals/monsters are 'action' or 'reveal'.\n"
            "   - 'action', 'combat', 'dialogue', 'transition'.\n"
            "4. FOCUS CHARACTER: Must be null OR a character from the active_characters list. Never use an unknown entity like 'Huge Fish'.\n"
            "5. IMPORTANCE: Rate each panel 1-10. 10 = epic/critical, 1 = minor filler.\n"
            "6. DESCRIPTION: Write a dense, visually descriptive prompt.\n"
            "7. NO MARKDOWN: Output ONLY valid JSON.\n\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "state": {\n'
            '    "current_location": "",\n'
            '    "time_of_day": "",\n'
            '    "weather": "",\n'
            '    "active_characters": []\n'
            "  },\n"
            '  "panels": [\n'
            "    {\n"
            '      "beat_type": "environment",\n'
            '      "shot_type": "wide_shot",\n'
            '      "importance": 8,\n'
            '      "location": "Riverbank",\n'
            '      "focus_character": "Arthur",\n'
            '      "characters": ["Arthur"],\n'
            '      "description": "Arthur sits beside the winding river under the glowing sunset."\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        payload = {
            "provided_state": current_state,
            "narrative_block": text_chunk
        }
        prompt = json.dumps(payload, indent=2)

        max_t = self.config.get("models", {}).get("llm", {}).get("scene_max_tokens", 3500)
        response = self.llm.generate_json(prompt, system_prompt=system, temperature=0.2, max_tokens=max_t)

        try:
            data = json.loads(response)
        except Exception as e:
            logger.warning(f"StoryboardPlanner JSON parse failed: {e}")
            return {"state": current_state, "panels": []}

        if not isinstance(data, dict):
            logger.warning("StoryboardPlanner output is not a dictionary.")
            return {"state": current_state, "panels": []}

        state = data.get("state", current_state)
        panels_raw = data.get("panels", [])
        
        if not isinstance(panels_raw, list):
            logger.warning("StoryboardPlanner 'panels' is not a list.")
            panels_raw = []

        valid_panels = []
        seq = start_sequence

        for p in panels_raw:
            try:
                bt = str(p.get("beat_type", "action")).lower()
                st = str(p.get("shot_type", "medium_shot")).lower()
                imp = int(p.get("importance", 5))
                desc = str(p.get("description", "")).strip()

                if bt not in self.VALID_BEATS:
                    bt = "action"
                if st not in self.VALID_SHOTS:
                    st = "medium_shot"
                if not (1 <= imp <= 10):
                    imp = max(1, min(10, imp))
                
                if not desc:
                    continue  # skip empty descriptions

                panel = {
                    "id": f"p{seq}",
                    "sequence": seq,
                    "beat_type": bt,
                    "shot_type": st,
                    "importance": imp,
                    "merge_with_previous": True if imp <= 2 else False,
                    "location": str(p.get("location", state.get("current_location", ""))),
                    "focus_character": str(p.get("focus_character", "")),
                    "characters": p.get("characters", []),
                    "description": desc,
                    "chapter": chapter
                }
                
                valid_panels.append(panel)
                seq += 1

            except Exception as e:
                logger.warning(f"Failed to validate panel {p}: {e}")
                continue

        return {"state": state, "panels": valid_panels}
