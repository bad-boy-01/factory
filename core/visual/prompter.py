"""
PromptGenerator — Novel Video Factory v5
Deterministically assembles Animagine-XL 3.1 prompts from beat panels.
"""
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ANIMAGINE_QUALITY = "score_9, score_8_up, (masterpiece:1.2), (best quality:1.1), highres"
MANHWA_STYLE = "manhwa, webtoon, korean manhwa style, (sharp lineart:1.2), cinematic composition"
MASTER_NEGATIVE = (
    "score_6, score_5, score_4, (worst quality, low quality:1.4), bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, "
    "signature, watermark, username, blurry, deformed, 3d render, photo, photorealistic, "
    "western comic, american comic, mutation"
)

_SHOT_TAGS = {
    "wide_shot": "wide shot, establishing shot, detailed background",
    "medium_shot": "medium shot, upper body",
    "close_up": "close-up portrait, face focus",
    "extreme_close_up": "extreme close-up, dramatic focus",
    "over_shoulder": "over-the-shoulder shot",
    "establishing_shot": "establishing shot, wide landscape, majestic",
}

class PromptGenerator:
    def __init__(self, memory_engine, config: dict = None, llm_adapter = None):
        self.memory = memory_engine
        self.config = config or {}
        self.project_dir = getattr(memory_engine, "project_dir", "")
        self.world_style = ""
        if self.project_dir:
            style_path = os.path.join(self.project_dir, "memory", "world_style.txt")
            if os.path.exists(style_path):
                with open(style_path, encoding="utf-8") as f:
                    self.world_style = f.read().strip()

    def generate_prompt_for_panel(self, panel: Dict, chapter_number: int = 1) -> Dict:
        """
        Build a prompt deterministically from the panel's metadata.
        """
        panel_id = panel.get("id", "p0")
        beat_type = panel.get("beat_type", "action")
        shot_type = panel.get("shot_type", "medium_shot")
        location = panel.get("location", "")
        focus_char = panel.get("focus_character", None)
        characters = panel.get("characters", [])
        description = panel.get("description", "")
        
        # 1. Shot framing
        shot_tag = _SHOT_TAGS.get(shot_type.lower(), "medium shot")
        
        # 2. Location DNA
        loc_tag = ""
        if location and location.lower() not in ["", "unknown"]:
            loc_data = self.memory.get_location_by_name(location)
            if loc_data:
                loc_tag = loc_data.get("visual_tags") or loc_data.get("description", "")

        # 3. Character DNA & Outfit
        char_tags = []
        ref_images = []
        
        # Bring focus character to the front
        sorted_chars = []
        if focus_char and focus_char in characters:
            sorted_chars.append(focus_char)
        for c in characters:
            if c != focus_char:
                sorted_chars.append(c)
                
        for char_name in sorted_chars:
            cdata = self.memory.get_character_by_name(char_name)
            if cdata:
                dna = cdata.get("visual_dna", {})
                outfit = cdata.get("current_state", {}).get("current_outfit", "")
                
                # Combine physical traits
                traits = []
                for k, v in dna.items():
                    if k in ["appearance_confidence", "role"]: continue
                    if isinstance(v, list):
                        traits.extend(str(x) for x in v if str(x).lower() not in ["none", "unknown"])
                    elif str(v).lower() not in ["none", "unknown", ""]:
                        traits.append(str(v))
                
                char_str = ", ".join(traits)
                if outfit:
                    char_str += f", wearing {outfit}"
                    
                # Visual lock string
                char_str += ", same character design, same face, consistent appearance"
                char_tags.append(f"({char_str})")
                
                # Reference image
                ref = self._find_reference(cdata["id"])
                if ref: ref_images.append(ref)

        # 4. Assemble
        parts = [
            ANIMAGINE_QUALITY,
            MANHWA_STYLE,
            shot_tag,
            self.world_style
        ]
        if loc_tag: parts.append(loc_tag)
        if char_tags: parts.extend(char_tags)
        parts.append(description)
        
        prompt = ", ".join(p.strip() for p in parts if p and p.strip())
        
        # Hash for cache
        cache_data = json.dumps({"p": prompt}, sort_keys=True)
        cache_key = hashlib.md5(cache_data.encode()).hexdigest()[:12]
        
        # Generation params
        importance = panel.get("importance", 5)
        steps = 25
        cfg = 7.0
        if importance >= 8:
            steps = 40
            cfg = 8.0
            
        img_cfg = self.config.get("models", {}).get("image", {})
        
        return {
            "scene_id": panel_id,
            "prompt": prompt,
            "negative_prompt": MASTER_NEGATIVE,
            "reference_images": list(set(ref_images)),
            "cache_key": cache_key,
            "generation_params": {
                "steps": steps,
                "cfg": cfg,
                "width": img_cfg.get("width", 832),
                "height": img_cfg.get("height", 480),
            }
        }

    def _find_reference(self, char_id: str) -> Optional[str]:
        if not self.project_dir: return None
        chars_dir = os.path.join(self.project_dir, "memory", "characters")
        legacy = os.path.join(chars_dir, f"{char_id}.png")
        if os.path.exists(legacy): return legacy
        pose = os.path.join(chars_dir, char_id, "front.png")
        if os.path.exists(pose): return pose
        return None
