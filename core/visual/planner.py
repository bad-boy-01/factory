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
            "YOU ARE NOT A STORYBOARD ARTIST.\n"
            "YOU ARE AN IMAGE BUDGET DIRECTOR.\n"
            "Your primary goal is to tell the story using the FEWEST possible images while preserving clarity.\n"
            "Think in VISUAL STATE CHANGES, not narrative beats.\n"
            "Turn the provided narrative block into a JSON storyboard.\n\n"
            "A major story realization does NOT automatically justify a new image.\n"
            "A new image is ONLY required when at least one of these changes:\n"
            "1. location\n"
            "2. character arrangement\n"
            "3. physical action\n"
            "4. visible object\n"
            "5. visible threat\n"
            "6. visual reveal\n"
            "7. camera framing requirement\n\n"
            "Everything else should reuse an existing image.\n"
            "The following MUST NOT create standalone images:\n"
            "- remembering, realizing, understanding, reflecting\n"
            "- internal monologue, emotional narration, backstory explanation, transmigration realization\n"
            "- Family reactions, worried expressions (reuse the existing family image).\n"
            "- Flashbacks (unless they occupy a significant portion of the chapter).\n"
            "These should be merged into the current image and represented through narration, subtitles, zooms, pans, and facial crops.\n\n"
            "LOCATION RULE: Every panel MUST contain the fully resolved location name. Do NOT output 'same as previous'.\n"
            "FOCUS CHARACTER RULE: focus_character must contain either null or a single character name. Never a comma-separated list.\n"
            "NARRATION TEST: If a panel description can be narrated over the previous image without visual confusion, force merge_with_previous=true.\n\n"
            "Always ask: 'Can this event be understood using the previous image?'\n"
            "If YES: merge_with_previous = true\n"
            "If uncertain: merge_with_previous = true\n"
            "Your default answer should always be MERGE.\n\n"
            "IMPORTANCE SCALE (1-10):\n"
            "   10 = Life-changing event, 9 = Major reveal/combat, 8 = Major action, 7 = Important story progression,\n"
            "   6 = Meaningful action, 5 = Useful visual context, 4 = Minor action, 3 = Minor reaction,\n"
            "   2 = Internal thought, 1 = Tiny detail.\n"
            "MAINTAIN CONTINUITY: Use the Provided State. Update the State based on the text.\n"
            "NO MARKDOWN: Output ONLY valid JSON.\n\n"
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

                merge = False
                if imp <= 4:
                    merge = True
                elif bt == "transition":
                    merge = True
                elif bt == "dialogue" and imp < 7:
                    merge = True
                elif bt == "reaction" and imp < 7:
                    merge = True
                elif bt == "emotion" and imp < 8:
                    merge = True

                raw_loc = str(p.get("location", ""))
                if not raw_loc or raw_loc.lower() in ["same", "same as previous", "current location", "unknown"]:
                    raw_loc = state.get("current_location", "")

                fc_raw = str(p.get("focus_character", "")).strip()
                if fc_raw.lower() in ["none", "null", ""]:
                    fc = None
                else:
                    fc = fc_raw.split(",")[0].strip()

                panel = {
                    "id": f"p{seq}",
                    "sequence": seq,
                    "beat_type": bt,
                    "shot_type": st,
                    "importance": imp,
                    "merge_with_previous": merge,
                    "location": raw_loc,
                    "focus_character": fc,
                    "characters": p.get("characters", []),
                    "description": desc,
                    "chapter": chapter
                }
                
                valid_panels.append(panel)
                seq += 1

            except Exception as e:
                logger.warning(f"Failed to validate panel {p}: {e}")
                continue

        # Post-processing pass: merge consecutive panels with same location and characters
        for i in range(1, len(valid_panels)):
            prev = valid_panels[i-1]
            curr = valid_panels[i]
            
            if curr["merge_with_previous"]:
                continue
                
            same_loc = (curr["location"] == prev["location"])
            same_chars = set(curr.get("characters", [])) == set(prev.get("characters", []))
            
            if same_loc and same_chars and curr["beat_type"] not in ["reveal", "combat"]:
                if curr["importance"] <= prev["importance"]:
                    curr["merge_with_previous"] = True
                else:
                    prev["merge_with_previous"] = True

        return {"state": state, "panels": valid_panels}
