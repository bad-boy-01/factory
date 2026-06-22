"""ProjectManager: Manages on-disk directory structure and checkpoints."""
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class ProjectManager:
    def __init__(self, base_dir: str, project_name: str):
        self.base_dir = base_dir
        self.project_name = project_name
        self.project_dir = os.path.join(base_dir, "projects", project_name)
        self.dirs = {
            "input":      os.path.join(self.project_dir, "input"),
            "output":     os.path.join(self.project_dir, "output"),
            "memory":     os.path.join(self.project_dir, "memory"),
            "storyboard": os.path.join(self.project_dir, "storyboard"),
            "export":     os.path.join(self.project_dir, "export"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.checkpoint_file = os.path.join(self.project_dir, "checkpoints.json")
        self._checkpoints = self._load_checkpoints()

    # ── Checkpoints ──────────────────────────────────────────────────────────
    def _load_checkpoints(self) -> dict:
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_checkpoints(self):
        tmp_file = self.checkpoint_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump(self._checkpoints, f, indent=2)
            os.replace(tmp_file, self.checkpoint_file)
        except Exception as e:
            logger.error(f"Failed to atomically save checkpoints: {e}")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def is_complete(self, stage: str, sub_key: str = "") -> bool:
        k = f"{stage}:{sub_key}" if sub_key else stage
        return self._checkpoints.get(k) is not None

    def save_checkpoint(self, stage: str, value, sub_key: str = ""):
        k = f"{stage}:{sub_key}" if sub_key else stage
        self._checkpoints[k] = value
        self._save_checkpoints()

    def get_checkpoint_value(self, stage: str, sub_key: str = ""):
        k = f"{stage}:{sub_key}" if sub_key else stage
        return self._checkpoints.get(k)

    # ── File I/O ──────────────────────────────────────────────────────────────
    def get_input_files(self) -> List[str]:
        exts = {".txt", ".md", ".epub"}
        return sorted(
            os.path.join(self.dirs["input"], f)
            for f in os.listdir(self.dirs["input"])
            if os.path.splitext(f)[1].lower() in exts
        )

    def read_input(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def save_output(self, filename: str, content: str) -> str:
        path = os.path.join(self.dirs["output"], filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
