"""CacheManager: Stage-level caching and prompt-image deduplication."""
import hashlib
import json
import logging
import os
import shutil
from typing import List

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, project_dir: str):
        self.cache_dir = os.path.join(project_dir, "cache")
        self.prompt_cache_dir = os.path.join(self.cache_dir, "prompts")
        os.makedirs(self.prompt_cache_dir, exist_ok=True)
        self.stage_file = os.path.join(self.cache_dir, "stages.json")
        self._stages = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.stage_file):
            try:
                with open(self.stage_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        with open(self.stage_file, "w") as f:
            json.dump(self._stages, f, indent=2)

    def _hash_files(self, paths: List[str]) -> str:
        h = hashlib.md5()
        for p in sorted(paths):
            if os.path.exists(p):
                h.update(str(os.path.getmtime(p)).encode())
        return h.hexdigest()

    def should_run_stage(self, stage: str, input_files: List[str]) -> bool:
        h = self._hash_files(input_files)
        return self._stages.get(stage) != h

    def mark_stage_complete(self, stage: str, input_files: List[str]):
        self._stages[stage] = self._hash_files(input_files)
        self._save()

    # ── Prompt-level image cache ───────────────────────────────────────────
    def apply_cache_hit(self, cache_key: str, target_path: str) -> bool:
        src = os.path.join(self.prompt_cache_dir, f"{cache_key}.png")
        if os.path.exists(src):
            shutil.copy2(src, target_path)
            logger.debug(f"Cache hit: {cache_key}")
            return True
        return False

    def store_cached_image(self, cache_key: str, source_path: str):
        if os.path.exists(source_path):
            dst = os.path.join(self.prompt_cache_dir, f"{cache_key}.png")
            shutil.copy2(source_path, dst)
