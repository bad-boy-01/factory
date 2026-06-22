"""Publishing Generator — generates YouTube metadata."""
import json
import logging
import os

logger = logging.getLogger(__name__)


class PublishingGenerator:
    def __init__(self, llm_adapter, project_dir: str, config: dict = None):
        self.llm = llm_adapter
        self.project_dir = project_dir
        self.config = config or {}

    def generate_seo_metadata(self, text_sample: str) -> dict:
        system = (
            "You are a YouTube SEO specialist for manhwa/anime content. "
            "Generate metadata for a manhwa novel adaptation video. "
            "Output ONLY a JSON object: "
            '{"title": "...", "description": "...", "tags": ["tag1", "tag2"]}'
        )
        result = self.llm.generate(text_sample, system_prompt=system, temperature=0.7)
        try:
            import re
            m = re.search(r"(\{.*\})", result, re.DOTALL)
            data = json.loads(m.group(1)) if m else {}
        except Exception:
            data = {"title": "Novel Adaptation", "description": "", "tags": []}

        path = os.path.join(self.project_dir, "output", "seo_metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"SEO metadata saved: {path}")
        return data
