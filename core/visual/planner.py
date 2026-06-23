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
            "STORYBOARDPLANNER MASTER DIRECTIVE\n\n"
            "You are a professional storyboard director responsible for converting prose into a sequence of images.\n"
            "Your goal is NOT to extract story beats.\n"
            "Your goal is to determine when the audience actually needs NEW VISUAL INFORMATION.\n\n"
            "CORE PRINCIPLE\n"
            "A new panel should only exist when the image on screen must visibly change.\n"
            "If the audience can understand the next event while still looking at the previous image, do not create a new image.\n"
            "Instead:\n"
            "* reuse the previous image\n"
            "* set merge_with_previous = true\n"
            "* allow narration, subtitles, camera motion, zooms, pans, and timing to carry the information\n\n"
            "VISUAL CHANGE TEST\n"
            "Before creating a panel ask:\n"
            "\"Would a human manga artist draw a completely new panel here?\"\n"
            "If NO: merge_with_previous = true\n"
            "If YES: create a new panel\n\n"
            "A NEW PANEL IS JUSTIFIED ONLY IF ONE OR MORE OF THESE CHANGE\n"
            "1. Location (e.g. room -> street, village -> forest)\n"
            "2. Character Arrangement (enter, leave, significant position change)\n"
            "3. Major Physical Action (attack, chase, fall, escape, transformation, combat)\n"
            "4. Important Visual Reveal (monster appears, hidden object discovered, secret identity revealed)\n"
            "5. Significant Environmental Change (day becomes night, weather changes, fire starts)\n"
            "6. Required Camera Reframing (establishing shot, close-up needed for important visual information)\n\n"
            "DO NOT CREATE NEW IMAGES FOR\n"
            "* thinking, remembering, realizing, understanding, internal monologue, narration, backstory explanation\n"
            "* emotional reflection, brief dialogue, minor reactions, looking around\n"
            "* walking without meaningful visual change, repeated views of the same scene\n\n"
            "FLASHBACK RULE\n"
            "Short memories never generate images. Memories are represented through facial expressions, narration, subtitles, camera movement. Only generate flashback images if it occupies a substantial portion of the chapter and contains unique visual events.\n\n"
            "EMOTION RULE\n"
            "Minor emotions never generate images (curiosity, concern, confusion, thinking, nostalgia, slight happiness). These must reuse existing images. Only create dedicated emotion panels for terror, rage, grief, despair, madness, overwhelming shock.\n\n"
            "REVEAL RULE\n"
            "A reveal must be visually observable. Valid: monster appears, hidden treasure found. Invalid: realizing something, learning information, discovering a fact mentally.\n\n"
            "OBJECT FOCUS RULE\n"
            "Object focus is reserved for important inanimate objects (sword, treasure, letter). Not valid: people, monsters, animals, facial expressions.\n\n"
            "VISUAL STATE UNIQUENESS RULE\n"
            "If two panels can be illustrated using the exact same image asset, they MUST collapse into a single panel. (e.g. 'Family surrounds bed' happens once. All subsequent dialogue/reactions reuse it).\n\n"
            "DUPLICATE ENVIRONMENT RULE\n"
            "Environment panels are only allowed when entering a new location, major environmental transformation, or major camera re-establishment. Environment may NOT be used for family reactions, dialogue, narration, reflection, memory recall, or exposition. Environment normally ONCE per location.\n\n"
            "MEMORY RULE\n"
            "A character remembering something does not create visual information. Never generate: 'Character remembers...', 'Character realizes...', 'Character recalls...', 'Character understands...'. These are narration events. They should be merged into the currently visible image.\n\n"
            "VISUAL DESCRIPTION RULE\n"
            "Descriptions must contain only things a camera can see.\n"
            "Forbidden: memories, realizations, understanding, internal thoughts, exposition.\n"
            "Allowed: faces, expressions, movement, posture, objects, environment, visible actions.\n\n"
            "IMAGE ECONOMY RULE\n"
            "One image should represent as much narrative as possible. (e.g. Character wakes up + family gathered + confusion = ONE image).\n\n"
            "FINAL QUESTION\n"
            "Before outputting any panel ask: \"If I remove this image entirely and keep the previous image on screen, can the audience still understand the story?\"\n"
            "If YES: merge_with_previous = true. If NO: create a new panel.\n\n"
            "STATE OVERRIDE: If the text introduces a completely new scene and characters, OVERRIDE the Provided State entirely. Do NOT carry over old locations or characters if they are not in the current text.\n"
            "LOCATION RULE: Every panel MUST contain the fully resolved location name. Do NOT output 'same as previous'.\n"
            "FOCUS CHARACTER RULE: focus_character must contain either null or a single character name. Never a comma-separated list.\n"
            "BEAT TYPES MUST BE USED CORRECTLY: environment, reveal, action, reaction, emotion, dialogue, transition, object_focus. Environment normally ONCE per location.\n"
            "IMPORTANCE SCALE (1-10):\n"
            "10=Life-changing, 9=Major reveal/combat, 8=Major action, 7=Important story progression, 6=Meaningful action, 5=Useful context, 4=Minor action, 3=Minor reaction, 2=Internal thought, 1=Tiny detail.\n"
            "Do NOT determine visual importance from beat_type. Determine visual importance from whether the audience receives new visual information.\n"
            "A realization is not visual. A visible transformation is visual. A dialogue line is not visual. A king publicly announcing war is visual.\n\n"
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

                import re
                internal_pattern = r"\b(realize|realizes|realization|remembers|remembered|memory|memories|thinks|thought|understands|understood|knows|learned|learns|transmigration|rebirth|past life|reflection|reflecting)\b"
                if re.search(internal_pattern, desc.lower()):
                    imp = min(imp, 4)
                    if bt in ["action", "combat", "reveal", "environment", "object_focus", "transition"]:
                        bt = "emotion"

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
        if valid_panels:
            last_unmerged_idx = 0
            for i in range(1, len(valid_panels)):
                curr = valid_panels[i]
                if curr["merge_with_previous"]:
                    continue
                    
                prev = valid_panels[last_unmerged_idx]
                
                same_loc = (curr["location"] == prev["location"])
                
                curr_chars = set(curr.get("characters", []))
                prev_chars = set(prev.get("characters", []))
                if curr_chars and prev_chars:
                    overlap = len(curr_chars.intersection(prev_chars)) / max(len(curr_chars), len(prev_chars))
                    same_chars = overlap >= 0.5 or curr_chars.issubset(prev_chars) or prev_chars.issubset(curr_chars)
                else:
                    same_chars = (curr_chars == prev_chars)
                
                is_env_change = curr["beat_type"] in ["reveal", "combat"] or (curr["beat_type"] == "action" and curr["importance"] >= 8)
                
                if same_loc and same_chars and not is_env_change:
                    curr["merge_with_previous"] = True
                else:
                    last_unmerged_idx = i

        return {"state": state, "panels": valid_panels}
