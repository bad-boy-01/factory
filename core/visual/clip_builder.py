"""
ClipBuilder — Novel Video Factory v4
Groups scenes into ~10-minute clips for Kaggle's session time limit.
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ClipBuilder:
    """
    Groups scenes into balanced 10-minute clips.
    This solves Kaggle's RAM/session limit: render one clip at a time.
    """
    def __init__(self, scenes_per_clip: int = 67):
        self.scenes_per_clip = scenes_per_clip

    def build_clips(self, scenes: List[Dict]) -> List[Dict]:
        """
        Group scenes into clips of scenes_per_clip each.
        Default: 67 scenes × ~9s/scene ≈ 10 minutes per clip.
        """
        if not scenes:
            return []

        total = len(scenes)
        clips = []
        clip_idx = 1

        for start in range(0, total, self.scenes_per_clip):
            end = min(start + self.scenes_per_clip, total)
            clip_scenes = scenes[start:end]
            clip_id = f"clip{clip_idx:02d}"

            # Re-index scene IDs to be clip-scoped
            for j, sc in enumerate(clip_scenes):
                sc["scene_id"] = f"{clip_id}_SC{j+1:03d}"
                # Deterministic seed from scene ID
                import hashlib
                sc["seed"] = (int(hashlib.sha256(sc["scene_id"].encode()).hexdigest()[:8], 16)
                              % (2**31 - 1))

            chapters = sorted({s.get("chapter", 1) for s in clip_scenes})
            clips.append({
                "clip_id": clip_id,
                "clip_index": clip_idx,
                "status": "planned",
                "chapters_covered": chapters,
                "scene_count": len(clip_scenes),
                "shots": clip_scenes,
            })
            clip_idx += 1

        logger.info(f"Grouped {total} scenes → {len(clips)} clips "
                    f"({self.scenes_per_clip} scenes/clip)")
        return clips
