"""
PromptGenerator — Novel Video Factory v4
Assembles Animagine-XL 3.1 prompts for Korean manhwa style.

Features:
- score_9 / score_8_up quality tokens (required for Animagine-XL)
- Character DNA injection with booru-style tags
- Emotion → reference pose selection (smile.png, angry.png, etc.)
- World style prefix for consistent art direction
- 16:9 optimised composition
"""
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Animagine-XL 3.1 quality prefix — MUST come first for best results
ANIMAGINE_QUALITY = "score_9, score_8_up, (masterpiece:1.2), (best quality:1.1), highres"

# Manhwa art style — consistent across all panels
MANHWA_STYLE = (
    "manhwa, webtoon, korean manhwa style, "
    "(sharp lineart:1.2), (vibrant colors:1.2), "
    "fully colored, digital illustration, "
    "cinematic composition, dynamic pose"
)

# Master negative for Animagine-XL (Anti-Watercolor Strategy)
MASTER_NEGATIVE = (
    "score_6, score_5, score_4, "
    "(worst quality, low quality:1.4), bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, "
    "cropped, jpeg artifacts, signature, watermark, username, "
    "blurry, ugly, deformed, 3d render, photo, photorealistic, "
    "western comic, american comic, fat, extra limbs, cloned face, "
    "mutation, fused fingers, long neck, retro, 90s anime, flat color, "
    "watercolor, oil painting, traditional media, painterly, brush strokes, "
    "textured paper, canvas, sketch, pencil, charcoal, western art style, "
    "comic, blurred, depth of field, bokeh, blurry"
)

# Emotion → best reference pose for IP-Adapter
_EMOTION_TO_POSE = {
    "neutral":  "front",
    "happy":    "smile",
    "angry":    "angry",
    "sad":      "crying",
    "fearful":  "sit",
    "fighting": "fight",
    "focused":  "sit",
    "shocked":  "front",
}
_POSE_FALLBACK = ["front", "sit", "smile", "angry", "crying", "fight"]

# Camera angle → composition tag
_CAMERA_TAGS = {
    "close-up":    "close-up, face focus, upper body",
    "medium shot": "medium shot, upper body, dynamic pose",
    "wide shot":   "wide shot, establishing shot, detailed background",
    "aerial":      "bird's eye view, aerial view, looking down",
    "low angle":   "low angle shot, looking up, dramatic perspective",
    "dutch angle": "dutch angle, tilted camera, dynamic tension",
}

# Safety margin under CLIP's hard 77-token-per-encoder cap.
# With Compel active, this is now a "soft" target to prevent prompts from 
# getting TOO long and slowing down generation, but we can safely go 
# much higher than 77.
PROMPT_TOKEN_BUDGET = 220


def _tag_list(text: str) -> List[str]:
    """Split a comma-separated tag string into a clean list, dropping blanks."""
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _approx_tokens(tag: str) -> int:
    """
    Conservative estimate of CLIP/BPE token count for one tag/phrase.
    Deliberately over-counts (extra weight for parens/colons used in
    Animagine-style emphasis syntax like "(masterpiece:1.2)") so this
    budget errs toward trimming a little early rather than letting a
    tag silently fall past the real 77-token edge.
    """
    n = len(tag.split())
    n += tag.count("(") + tag.count(")") + tag.count(":")
    return max(1, n)


def _dedup_tags(*groups: List[str]) -> List[str]:
    """
    Flatten tag groups in priority order, dropping case-insensitive
    duplicates. Fixes the (real, observed) bug where a misconfigured
    style_positive re-included the same score_9/masterpiece/best-quality
    tags already present in ANIMAGINE_QUALITY, wasting ~15 tokens of the
    77-token budget on repeats before a single character or style tag
    was added.
    """
    seen = set()
    out = []
    for group in groups:
        for tag in group:
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                out.append(tag)
    return out


def _apply_budget(tags: List[str], budget: int) -> List[str]:
    """
    Keep tags — already given in priority order — until the token budget
    is used up, then stop. Stopping (rather than skipping a too-big tag
    and grabbing a smaller one further down) keeps degradation predictable:
    if something has to go, it's always the lowest-priority tail, never an
    arbitrary tag picked because it happened to be cheap.
    """
    out, used = [], 0
    for tag in tags:
        cost = _approx_tokens(tag)
        if used + cost > budget:
            break
        out.append(tag)
        used += cost
    return out


class PromptGenerator:
    def __init__(self, memory_engine, config: dict = None, llm_adapter = None):
        self.memory = memory_engine
        self.config = config or {}
        self.llm = llm_adapter
        self.project_dir = getattr(memory_engine, "project_dir", "")

        # Load world style set during memory extraction
        self.world_style = ""
        if self.project_dir:
            style_path = os.path.join(self.project_dir, "memory", "world_style.txt")
            if os.path.exists(style_path):
                with open(style_path, encoding="utf-8") as f:
                    self.world_style = f.read().strip()

        prompts_cfg = self.config.get("prompts", {})
        self.style_positive = prompts_cfg.get("style_positive", MANHWA_STYLE) or MANHWA_STYLE
        raw_negative = prompts_cfg.get("style_negative", MASTER_NEGATIVE) or MASTER_NEGATIVE
        # Dedup (MASTER_NEGATIVE has "blurry" listed twice) and apply the same
        # token budget — the negative prompt is tokenized through the same
        # 77-token CLIP encoders as the positive one.
        self.style_negative = ", ".join(
            _apply_budget(_dedup_tags(_tag_list(raw_negative)), PROMPT_TOKEN_BUDGET)
        )

    def rewrite_prompt(self, original_prompt: str, scene_action: str) -> str:
        """
        Rewrites a prompt that failed to generate a high-quality image.
        Simplifies and clarifies tags using the LLM.
        """
        if self.llm is None:
            return original_prompt

        system = (
            "You are a prompt engineer for SDXL/Animagine. "
            "The following prompt failed to produce a good image. "
            "Rewrite it to be clearer and more effective. "
            "RULES: "
            "1. Keep the core characters and setting. "
            "2. Simplify complex actions into clear visual tags. "
            "3. Ensure Animagine quality tags are present at the start. "
            "4. Output ONLY the improved comma-separated tags."
        )
        prompt = f"Original Prompt: {original_prompt}\nScene Action: {scene_action}"
        try:
            return self.llm.generate(prompt, system_prompt=system, temperature=0.3)
        except Exception:
            return original_prompt

    def generate_prompt_for_scene(self, scene: Dict, chapter_number: int = 1) -> Dict:
        """
        Build a complete Animagine-XL 3.1 prompt for one manhwa panel.
        Returns dict with: scene_id, prompt, negative_prompt, metadata,
                           reference_images, cache_key, generation_params
        """
        characters: List[str] = scene.get("characters", [])
        location_name: str = scene.get("location", "") or ""
        emotion: str = scene.get("emotion", "neutral")
        action_tags: str = scene.get("visual_prompt_tags", "")
        camera_raw: str = scene.get("camera_angle", "medium shot")
        lighting: str = scene.get("lighting", "cinematic lighting")

        # 1. Camera composition
        camera_tags = _CAMERA_TAGS.get(camera_raw.lower(), camera_raw)

        # 2. Character DNA
        char_parts: List[str] = []  # one chunk per character, e.g. "(1boy, black hair, ...)"
        ref_images: List[str] = []
        char_cache_data: List[tuple] = []

        n = len(characters)
        if n == 1:
            count_tag = "1person, solo"
        elif n == 2:
            count_tag = "2people"
        elif n >= 3:
            count_tag = f"{n}people, group"
        else:
            count_tag = ""

        for char_name in characters:
            char_data = self.memory.get_character_by_name(char_name)
            if char_data:
                dna = char_data.get("visual_dna", {})
                current_state = char_data.get("current_state", {})

                # Filter out the subject/gender tag from DNA — we add it explicitly
                # to avoid duplicates like "(1boy, 1boy, black hair...)"
                GENDER_TAGS = {"1boy", "1girl", "1man", "1woman", "boy", "girl", "male", "female"}

                # Keys that are bookkeeping/narrative metadata, never visual
                # descriptors. Previously this loop did `str(v) for v in
                # dna.values()` over every field with NO key filtering, so
                # appearance_confidence (e.g. 0.9) and role (e.g. "Xu
                # Changshou's mother" — a relationship label, not what she
                # looks like) got tokenized straight into the prompt as if
                # they were visual tags. "importance" lives on current_state,
                # not dna, but is excluded here too for the same reason.
                NON_VISUAL_DNA_KEYS = {"appearance_confidence", "role"}
                NON_VISUAL_STATE_KEYS = {"emotion", "importance"}
                EMPTY_VALUES = {"", "none", "unknown", "not specified", "normal"} | GENDER_TAGS

                def _flatten_field(value) -> List[str]:
                    """
                    Turn one DNA/state field into zero or more clean tag
                    strings. List fields (e.g. distinctive_features) used to
                    be passed through Python's str() on the whole list,
                    which produces literal repr text like "['memories of a
                    past life', 'innocent expression']" — brackets, quotes,
                    commas and all — directly in the prompt. Each list item
                    is now its own separate tag instead.
                    """
                    if isinstance(value, (list, tuple, set)):
                        out = []
                        for item in value:
                            s = str(item).strip()
                            if s and s.lower() not in EMPTY_VALUES:
                                out.append(s)
                        return out
                    s = str(value).strip()
                    return [s] if s and s.lower() not in EMPTY_VALUES else []

                dna_tags: List[str] = []
                for k, v in dna.items():
                    if k in NON_VISUAL_DNA_KEYS or not v:
                        continue
                    dna_tags.extend(_flatten_field(v))

                # Inject dynamic state (outfit overrides, injuries)
                for k, v in current_state.items():
                    if k in NON_VISUAL_STATE_KEYS or not v:
                        continue
                    dna_tags.extend(_flatten_field(v))

                dna_str = ", ".join(dna_tags)
                # Detect gender from full DNA dict values (including filtered ones)
                all_dna_vals = " ".join(str(v).lower() for v in dna.values())
                gender = ("1girl"
                          if any(w in all_dna_vals for w in ["girl", "woman", "female", "she"])
                          else "1boy")
                char_parts.append(f"({gender}, {dna_str})" if dna_str else gender)
                
                # Use dynamic emotion if static parameter is neutral
                scene_emotion = emotion
                if scene_emotion == "neutral" and current_state.get("emotion"):
                    scene_emotion = str(current_state.get("emotion")).lower()
                    
                ref = self._find_reference(char_data["id"], scene_emotion)
                if ref:
                    ref_images.append(ref)
                
                # Cache key must include state changes
                char_cache_data.append((char_data["id"], json.dumps(dna, sort_keys=True), json.dumps(current_state, sort_keys=True)))
            else:
                char_parts.append("1boy")

        # 3. Location context
        location_tags = ""
        if location_name and location_name.lower() not in {"", "unknown", "unknown location"}:
            loc = self.memory.get_location_by_name(location_name)
            if loc:
                raw = loc.get("visual_tags") or loc.get("description") or ""
                if raw:
                    location_tags = ", ".join(t.strip() for t in raw.split(",")[:6])

        # 4. Assemble prompt.
        # Tier 1 (reserved, never trimmed): quality + the actual manhwa/style
        # tags, deduped against each other. This is the fix for the #1
        # observed bug — style_positive used to be appended dead last, after
        # character/location/world tags that often already ate the full
        # 77-token budget, so "korean manhwa style" was the first thing CLIP
        # silently dropped on nearly every single image.
        header_tags = _dedup_tags(_tag_list(ANIMAGINE_QUALITY), _tag_list(self.style_positive))
        reserved_cost = sum(_approx_tokens(t) for t in header_tags)
        remaining_budget = max(10, PROMPT_TOKEN_BUDGET - reserved_cost)

        # Tier 2 (budgeted, priority order): everything else.
        #
        # Characters get special handling: every character in the scene
        # gets at least a bare presence tag ("1boy"/"1girl") guaranteed, and
        # is only *upgraded* to their full DNA group (in scene order) if
        # there's still room. Previously, once the budget ran out partway
        # through a multi-character scene, the remaining characters got no
        # representation in the prompt at all — in a 6-person scene this
        # meant characters #3–6 were invisible to the model entirely, not
        # just under-described. A character's whole group is still kept
        # atomic (never split mid-attribute) to avoid attribute bleed.
        bare_tags, full_tags = [], []
        for part in char_parts:
            if part.startswith("(") and "," in part:
                bare_tags.append(part[1:].split(",", 1)[0].strip())
            else:
                bare_tags.append(part.strip("()"))
            full_tags.append(part)

        chosen = list(bare_tags)
        # NOTE: deliberately NOT deduped — two different characters can
        # legitimately both reduce to a bare "1girl" (e.g. two undescribed
        # women in the same scene), and collapsing those into one shared
        # tag would silently erase one of them from the prompt entirely.
        # Dedup only ever applies to header/style/location tags below,
        # never to the list of per-character tags itself.
        used = sum(_approx_tokens(t) for t in chosen)
        used += _approx_tokens(count_tag) if count_tag else 0
        for i, full in enumerate(full_tags):
            if full == chosen[i]:
                continue
            delta = _approx_tokens(full) - _approx_tokens(chosen[i])
            if used + delta <= remaining_budget:
                chosen[i] = full
                used += delta

        char_section = ([count_tag] if count_tag else []) + chosen
        rest_pool = _dedup_tags(_tag_list(action_tags), _tag_list(camera_tags),
                                 _tag_list(lighting), _tag_list(location_tags),
                                 _tag_list(self.world_style))
        rest = _apply_budget(rest_pool, max(0, remaining_budget - used))
        body_tags = char_section + rest

        final_tags = header_tags + body_tags
        prompt = ", ".join(final_tags)

        downgraded = sum(1 for c, f in zip(chosen, full_tags) if c != f)
        dropped = len(rest_pool) - len(rest)
        if downgraded or dropped:
            logger.warning(
                f"  Prompt budget for scene {scene.get('scene_id', '?')}: "
                f"{downgraded} character(s) kept as bare tags (no DNA) and "
                f"{dropped} action/location/style tag(s) dropped to stay under "
                f"{PROMPT_TOKEN_BUDGET} tokens. Quality/manhwa-style tags were "
                f"never at risk."
            )

        char_prompt = ", ".join(char_parts)  # kept for cache_data below

        # 5. Cache key for deduplication
        cache_data = json.dumps({
            "chars": sorted(char_cache_data),
            "location": location_name,
            "action": action_tags,
            "world": self.world_style,
        }, sort_keys=True)
        cache_key = hashlib.md5(cache_data.encode()).hexdigest()[:12]

        img_cfg = self.config.get("models", {}).get("image", {})
        return {
            "scene_id": scene.get("scene_id", "SC000"),
            "prompt": prompt,
            "negative_prompt": self.style_negative,
            "metadata": scene,
            "reference_images": list(set(ref_images)),
            "cache_key": cache_key,
            "generation_params": {
                "steps": img_cfg.get("num_inference_steps", 20),
                "cfg": img_cfg.get("guidance_scale", 7.0),
                "width": img_cfg.get("width", 832),
                "height": img_cfg.get("height", 480),
            },
        }

    def _find_reference(self, char_id: str, emotion: str) -> Optional[str]:
        """Find the best matching reference pose image for IP-Adapter."""
        if not self.project_dir:
            return None
        chars_dir = os.path.join(self.project_dir, "memory", "characters")
        preferred_pose = _EMOTION_TO_POSE.get(emotion, "front")

        # Multi-pose subfolder: memory/characters/{id}/{pose}.png
        pose_dir = os.path.join(chars_dir, char_id)
        if os.path.isdir(pose_dir):
            for pose in [preferred_pose] + _POSE_FALLBACK:
                candidate = os.path.join(pose_dir, f"{pose}.png")
                if os.path.exists(candidate):
                    return candidate

        # Legacy: single image memory/characters/{id}.png
        legacy = os.path.join(chars_dir, f"{char_id}.png")
        if os.path.exists(legacy):
            return legacy

        return None
