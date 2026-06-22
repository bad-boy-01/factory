"""
MemoryExtractor — Novel Video Factory v4
LLM-based extraction of story entities for the knowledge store.
"""
import json
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class MemoryExtractor:
    def __init__(self, llm_adapter, config: dict = None):
        self.llm = llm_adapter
        self.config = config or {}

    def _gen(self, prompt: str, system: str, temperature: float = 0.1) -> str:
        return self.llm.generate(prompt, system_prompt=system, temperature=temperature)

    def _gen_json(self, prompt: str, system: str, temperature: float = 0.1) -> str:
        return self.llm.generate_json(prompt, system_prompt=system, temperature=temperature)

    def extract_all(self, text: str, existing_characters: list = None) -> Dict:
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or []) if c.get("canonical_name")]
        relevant_names = [n for n in existing_names if n.lower() in text.lower()]
        known = ", ".join(relevant_names) if relevant_names else "none yet"

        system = (
            "You are a master story extraction engine.\n\n"
            "Extract characters, locations, events, and world style from the text.\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- No markdown, no explanations.\n"
            "- Include every named character.\n"
            "- One event per significant action, preserve chronological order.\n"
            "- For world_style, provide 10-15 comma-separated booru tags describing era, architecture, climate, and atmosphere.\n"
            f"- Known characters to look for: {known}\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "characters": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "importance": 5,\n'
            '      "age": "",\n'
            '      "gender": "",\n'
            '      "hair": "",\n'
            '      "eyes": "",\n'
            '      "face": "",\n'
            '      "build": "",\n'
            '      "clothing": "",\n'
            '      "accessories": "",\n'
            '      "distinctive_features": [],\n'
            '      "appearance_confidence": 0.0,\n'
            '      "role": ""\n'
            "    }\n"
            "  ],\n"
            '  "locations": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "description": ""\n'
            "    }\n"
            "  ],\n"
            '  "events": [\n'
            "    {\n"
            '      "event_id": 1,\n'
            '      "summary": "",\n'
            '      "importance": 8,\n'
            '      "characters": []\n'
            "    }\n"
            "  ],\n"
            '  "world_style": ""\n'
            "}"
        )
        prompt = f"TEXT:\n\n{text[:2500]}"
        logger.info(f"  [Compression] Unified Extraction. Prompt length: {len(prompt)} chars.")

        max_t = self.config.get("models", {}).get("llm", {}).get("event_max_tokens", 3000)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.0, max_tokens=max_t)
        
        if "_quota_exhausted" in result:
            return {"_parse_error": True, "_quota_exhausted": True, "_raw_text": result}
            
        try:
            parsed = json.loads(result)
            parsed["_parse_error"] = False
            if "events" in parsed and isinstance(parsed["events"], list):
                clean_events = []
                for e in parsed["events"]:
                    if isinstance(e, dict):
                        clean_events.append(e)
                    elif isinstance(e, str):
                        clean_events.append({
                            "summary": e,
                            "involved_characters": [],
                            "location": "",
                            "description": ""
                        })
                parsed["events"] = clean_events
            return parsed
        except Exception as e:
            logger.warning(f"extract_all JSON parse failed: {e}")
            return {"_parse_error": True, "characters": [], "locations": [], "events": [], "world_style": "", "_raw_text": result}

    def extract_characters(self, text: str, existing_characters: list = None) -> Dict:
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or []) if c.get("canonical_name")]
        relevant_names = [n for n in existing_names if n.lower() in text.lower()]
        known = ", ".join(relevant_names) if relevant_names else "none yet"

        system = (
            "You are an information extraction engine.\n\n"
            "Extract ONLY characters mentioned in the text.\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON.\n"
            "- No markdown.\n"
            "- No explanations.\n"
            "- Include every named character.\n"
            "- Do not invent characters.\n"
            '- If no characters exist, return {"characters":[]}\n\n'
            "JSON Schema:\n"
            "{\n"
            '  "characters": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "importance": 5,\n'
            '      "age": "",\n'
            '      "gender": "",\n'
            '      "hair": "",\n'
            '      "eyes": "",\n'
            '      "face": "",\n'
            '      "build": "",\n'
            '      "clothing": "",\n'
            '      "accessories": "",\n'
            '      "distinctive_features": [],\n'
            '      "appearance_confidence": 0.0,\n'
            '      "role": ""\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        prompt = f"TEXT:\n\n{text[:2500]}"
        logger.info(f"  [Compression] Characters: {len(existing_names)} total -> {len(relevant_names)} relevant. Prompt length: {len(prompt)} chars.")
        
        max_t = self.config.get("models", {}).get("llm", {}).get("character_max_tokens", 1200)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.0, max_tokens=max_t)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_characters JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

    def extract_locations(self, text: str) -> Dict:
        system = (
            "You are an information extraction engine.\n\n"
            "Extract all locations mentioned in the text.\n\n"
            "Rules:\n"
            "- Return ONLY JSON.\n"
            "- No markdown.\n"
            "- No explanations.\n"
            "- Do not invent locations.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "locations": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "description": ""\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        prompt = f"TEXT:\n\n{text[:2500]}"
        max_t = self.config.get("models", {}).get("llm", {}).get("location_max_tokens", 800)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.0, max_tokens=max_t)
        try:
            return json.loads(result)
        except Exception as e:
            logger.warning(f"extract_locations JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

    def extract_events(self, text: str, existing_characters: list = None) -> Dict:
        existing_names = [c.get("canonical_name", "") for c in (existing_characters or []) if c.get("canonical_name")]
        relevant_names = [n for n in existing_names if n.lower() in text.lower()]
        known = ", ".join(relevant_names) if relevant_names else "none yet"

        system = (
            "You are extracting story events.\n\n"
            "Extract major story beats in chronological order.\n\n"
            "Rules:\n"
            "- Return ONLY JSON.\n"
            "- No markdown.\n"
            "- No explanations.\n"
            "- One event per significant action.\n"
            "- Preserve order.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "events": [\n'
            "    {\n"
            '      "event_id": 1,\n'
            '      "summary": "",\n'
            '      "importance": 8,\n'
            '      "characters": []\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        prompt = f"TEXT:\n\n{text[:2500]}"
        logger.info(f"  [Compression] Events: {len(existing_names)} total -> {len(relevant_names)} relevant. Prompt length: {len(prompt)} chars.")

        max_t = self.config.get("models", {}).get("llm", {}).get("event_max_tokens", 2000)
        result = self.llm.generate_json(prompt, system_prompt=system, temperature=0.0, max_tokens=max_t)
        try:
            parsed = json.loads(result)
            if "events" in parsed and isinstance(parsed["events"], list):
                clean_events = []
                for e in parsed["events"]:
                    if isinstance(e, dict):
                        clean_events.append(e)
                    elif isinstance(e, str):
                        clean_events.append({
                            "summary": e,
                            "involved_characters": [],
                            "location": "",
                            "description": ""
                        })
                parsed["events"] = clean_events
            return parsed
        except Exception as e:
            logger.warning(f"extract_events JSON parse failed: {e}")
            return {"_parse_error": True, "_raw_text": result}

    def extract_world_style(self, text: str) -> str:
        """Extract a short visual style string for consistent art direction."""
        system = (
            "You are a visual art director for manhwa. "
            "Describe the visual aesthetic of this world in 10-15 comma-separated booru tags. "
            "Focus on: era/time period, architecture style, climate, color palette, atmosphere. "
            "Examples: 'ancient China, stone pagodas, misty mountains, imperial colors' "
            "or 'modern Seoul, neon lights, urban fantasy, rainy streets'. "
            "Output ONLY the comma-separated tags, nothing else."
        )
        result = self._gen(text[:1000], system)
        tags = result.strip().strip("`").split("\n")[0].strip()
        return tags[:300]
