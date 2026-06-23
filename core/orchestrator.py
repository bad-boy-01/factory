
"""
UnifiedPipeline — Novel Video Factory v5

Coordinates all pipeline stages:
  1. translate   → convert non-English text to English (smart skip for English)
  2. memory      → extract characters, locations, lore from text
  3. char_sheets → generate multi-pose character reference images
  4. visual      → plan scenes and build prompts
  5. generation  → generate manhwa images
  6. audio       → generate TTS narration
  7. video       → assemble 10-minute clips
  8. export      → stitch all clips into final 2-hour video
"""
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def _canonicalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s+", " ", name).strip()
    lower_name = name.lower()
    for article in ("the ", "a ", "an "):
        if lower_name.startswith(article):
            return name[len(article):].strip()
    return name

def _chunk_text(text: str, max_words: int = 500) -> list:
    """Split text into chunks using SmartChunker while maintaining legacy dict format."""
    from core.utils.chunker import SmartChunker
    chunker = SmartChunker(target_words=int(max_words*0.8), max_words=max_words)
    raw_chunks = chunker.chunk_text(text)
    
    formatted_chunks = []
    global_sent_idx = 0
    
    for chunk_str in raw_chunks:
        # Re-split into sentences for the legacy format
        pattern = r"(?<=[.!?。！？\n])\s+"
        sentences = [s.strip() for s in re.split(pattern, chunk_str.strip()) if s.strip()]
        
        current_sents = []
        current_words = 0
        for s in sentences:
            wc = len(s.split())
            current_sents.append({"index": global_sent_idx, "text": s})
            current_words += wc
            global_sent_idx += 1
            
        formatted_chunks.append({
            "sentences": current_sents,
            "word_count": current_words
        })
        
    return formatted_chunks


class UnifiedPipeline:
    """
    End-to-end novel-to-manhwa-video pipeline with VRAM protection.
    """

    def __init__(self, project_name: str, config_path: str = "config/default.yaml"):
        from core.config_manager import ConfigManager
        from core.project_manager import ProjectManager
        from core.memory.database import MemoryEngine

        self.project_name = project_name
        self.base_dir = os.getcwd()
        self.config = ConfigManager(config_path)
        self.pm = ProjectManager(self.base_dir, project_name)
        self.memory_db = MemoryEngine(self.pm.project_dir)

        self._extractor_llm = None
        self._planner_llm = None
        self._image_gen = None
        self._audio_gen = None
        
        self.extractor_unavailable = False

        self.metrics = {
            "chunks_processed": 0,
            "chunks_failed": 0,
            "character_success": 0,
            "location_success": 0,
            "event_success": 0,
            "llm_failures": 0,
            "retries_queued": 0,
            "avg_extraction_latency": 0.0,
            "avg_planning_latency": 0.0,
            "extractor_quota_exhausted": False,
            "extractor_provider": self.config.get("models", {}).get("llm", {}).get("extractor_provider", "gemini"),
            "planner_provider": self.config.get("models", {}).get("llm", {}).get("planner_provider", "groq")
        }

        logger.info(f"Pipeline ready for project: {project_name}")

    def _archive_raw_response(self, stage: str, chunk_id: str, raw_text: str):
        raw_dir = os.path.join(self.pm.dirs["output"], "raw_llm")
        os.makedirs(raw_dir, exist_ok=True)
        filename = f"{chunk_id}_{stage}_raw.txt"
        with open(os.path.join(raw_dir, filename), "w", encoding="utf-8") as f:
            f.write(str(raw_text))

    def _add_to_retry_queue(self, stage: str, chunk_id: str, reason: str):
        queue_path = os.path.join(self.pm.dirs["output"], "retry_queue.json")
        queue = []
        if os.path.exists(queue_path):
            with open(queue_path, "r", encoding="utf-8") as f:
                queue = json.load(f)
        
        # Check if already in queue
        for item in queue:
            if item["chunk_id"] == chunk_id and item["stage"] == stage:
                item["attempts"] += 1
                item["reason"] = reason
                break
        else:
            queue.append({
                "chunk_id": chunk_id,
                "stage": stage,
                "reason": reason,
                "attempts": 1
            })
            
        with open(queue_path, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)
            
        self.metrics["retries_queued"] += 1

    def _save_metrics(self):
        metrics_path = os.path.join(self.pm.project_dir, "artifacts", "run_metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

    # ── Lazy adapters ─────────────────────────────────────────────────────────
    @property
    def extractor_llm(self):
        if self._extractor_llm is None:
            from models.llm_adapter import SmartLLMAdapter
            ext_provider = self.config.get("models", {}).get("llm", {}).get("extractor_provider", "gemini")
            allow_fallback = self.config.get("models", {}).get("llm", {}).get("extractor_allow_fallback", False)
            self._extractor_llm = SmartLLMAdapter(config=self.config.config, provider_override=ext_provider, allow_fallback=allow_fallback)
        return self._extractor_llm

    @property
    def planner_llm(self):
        if self._planner_llm is None:
            from models.llm_adapter import SmartLLMAdapter
            plan_provider = self.config.get("models", {}).get("llm", {}).get("planner_provider", "groq")
            allow_fallback = self.config.get("models", {}).get("llm", {}).get("planner_allow_fallback", True)
            self._planner_llm = SmartLLMAdapter(config=self.config.config, provider_override=plan_provider, allow_fallback=allow_fallback)
        return self._planner_llm

    @property
    def image_gen(self):
        if self._image_gen is None:
            self._unload_llm()
            from models.image_adapter import LocalImageAdapter
            self._image_gen = LocalImageAdapter(config=self.config.config)
        return self._image_gen

    @property
    def audio_gen(self):
        if self._audio_gen is None:
            from models.audio_adapter import LocalAudioAdapter
            self._audio_gen = LocalAudioAdapter(config=self.config.config)
        return self._audio_gen

    def _unload_llm(self):
        if self._extractor_llm is not None:
            self._extractor_llm.unload_model()
            self._extractor_llm = None
        if self._planner_llm is not None:
            self._planner_llm.unload_model()
            self._planner_llm = None
            gc.collect()

    def _unload_image_gen(self):
        if self._image_gen is not None:
            self._image_gen.unload()
            self._image_gen = None
            gc.collect()

    # ── Run All ───────────────────────────────────────────────────────────────
    def run_all(self, source_file: Optional[str] = None):
        """Run the entire pipeline from start to finish."""
        if source_file:
            self.import_source(source_file)
            
        if not getattr(self.extractor_llm, "is_available", True):
            logger.error("🛑 CRITICAL: Extractor LLM provider is unreachable.")
            return

        if not getattr(self.planner_llm, "is_available", True):
            logger.error("🛑 CRITICAL: Planner LLM provider is unreachable.")
            return
            
        self.stage_translate()
        self.stage_memory()
        
        if self.metrics.get("extractor_quota_exhausted"):
            logger.error("🛑 CRITICAL: Memory extraction incomplete due to LLM quota exhaustion.")
            logger.error("🛑 Pipeline aborted to prevent generating severely degraded output.")
            return
            
        self.stage_character_sheets()
        self.stage_visual_planning()
        self.stage_generation()
        self.stage_audio()
        self.stage_video()
        self.stage_export()
        self._save_metrics()
        logger.info("✅ Full pipeline complete!")

    def import_source(self, source_file: str):
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Input file not found: {source_file}")
        dest = os.path.join(self.pm.dirs["input"], os.path.basename(source_file))
        
        # Avoid SameFileError if source is already in the project's input folder
        if os.path.abspath(source_file) == os.path.abspath(dest):
            logger.info(f"Source already in project input: {dest}")
            return

        shutil.copy(source_file, dest)
        logger.info(f"Imported: {source_file} → {dest}")

    # ── Stage 1: Translation ──────────────────────────────────────────────────
    def stage_translate(self):
        logger.info("─── Stage 1: Translation ───")
        from core.translation.pipeline import TranslationPipeline

        input_files = self.pm.get_input_files()
        if not input_files:
            logger.warning(f"No input files found in projects/{self.project_name}/input/")
            return

        pipeline = TranslationPipeline(config=self.config.config, llm_adapter=self.extractor_llm)

        for file_path in input_files:
            filename = os.path.basename(file_path)
            out_name = f"translated_{filename}"
            out_path = os.path.join(self.pm.dirs["output"], out_name)

            if os.path.exists(out_path) and not self.pm.is_complete("translate", filename):
                # Output exists but no checkpoint — mark as done
                self.pm.save_checkpoint("translate", "done", sub_key=filename)

            if self.pm.is_complete("translate", filename):
                logger.info(f"  Skipping (cached): {filename}")
                continue

            text = self.pm.read_input(file_path)
            
            if pipeline.is_translation_needed(text):
                if not getattr(self.extractor_llm, "is_available", True):
                    logger.error(f"❌ Stage 1 Aborted: Translation required for {filename} but no LLM provider is reachable.")
                    return
            
            translated = pipeline.process_chapter(text)
            self.pm.save_output(out_name, translated)
            self.pm.save_checkpoint("translate", "done", sub_key=filename)
            logger.info(f"  ✓ Translated: {filename}")

        logger.info("✅ Translation complete")

    # ── Stage 2: Memory Extraction ────────────────────────────────────────────
    def stage_memory(self):
        logger.info("─── Stage 2: Memory Extraction ───")
        if not getattr(self.extractor_llm, "is_available", True):
            logger.error("❌ Stage 2 Aborted: No LLM provider reachable (Groq/Ollama). "
                         "Check your internet or start Ollama local server.")
            return

        from core.memory.extractor import MemoryExtractor

        translated_files = sorted(
            f for f in os.listdir(self.pm.dirs["output"])
            if f.startswith("translated_")
        )
        if not translated_files:
            logger.warning("No translated files found — run stage_translate first")
            return

        extractor = MemoryExtractor(self.extractor_llm, config=self.config.config)
        world_style_saved = False

        for filename in translated_files:
            file_path = os.path.join(self.pm.dirs["output"], filename)
            text = self.pm.read_input(file_path)
            chunks = _chunk_text(text, max_words=750)

            for idx, chunk_data in enumerate(chunks):
                sub_key = f"mem_{filename}_{idx}"
                if self.pm.is_complete("memory", sub_key):
                    continue

                chunk_text = " ".join(s["text"] for s in chunk_data["sentences"])
                
                if self.extractor_unavailable:
                    logger.info(f"  ⏭️ Skipping chunk {idx+1}/{len(chunks)} from {filename} (quota exhausted)")
                    self._add_to_retry_queue("memory", sub_key, "quota_exhausted")
                    continue

                logger.info(f"  Extracting chunk {idx+1}/{len(chunks)} from {filename}")

                # Proactive delay to avoid Groq Rate Limits
                import time
                if getattr(self.extractor_llm, "is_cloud", False):
                    time.sleep(5)

                existing_chars = self.memory_db.get_all_characters()
                
                import time
                start_time = time.time()
                
                try:
                    data = extractor.extract_all(chunk_text, existing_characters=existing_chars)
                    
                    if data.get("_quota_exhausted"):
                        self.metrics["chunks_failed"] += 1
                        self.metrics["extractor_quota_exhausted"] = True
                        self.extractor_unavailable = True
                        logger.warning(f"  ⚠️  Extractor quota exhausted on Chunk {idx+1}. Remaining chunks queued.")
                        self._add_to_retry_queue("memory", sub_key, "quota_exhausted")
                        continue
                        
                    if data.get("_parse_error"):
                        self.metrics["chunks_failed"] += 1
                        self._archive_raw_response("unified", sub_key, data.get("_raw_text", ""))
                        self._add_to_retry_queue("memory", sub_key, "parse_failed")
                        continue
                        
                    self.metrics["character_success"] += 1 if data.get("characters") else 0
                    self.metrics["location_success"] += 1 if data.get("locations") else 0
                    self.metrics["event_success"] += 1 if data.get("events") else 0
                    
                    elapsed = time.time() - start_time
                    prev_avg = self.metrics["avg_extraction_latency"]
                    n = self.metrics["chunks_processed"]
                    self.metrics["avg_extraction_latency"] = (prev_avg * n + elapsed) / (n + 1)
                    
                    self.metrics["chunks_processed"] += 1

                    if not data.get("characters") and not data.get("locations") and not data.get("events"):
                        self.metrics["chunks_failed"] += 1
                        self._add_to_retry_queue("memory", sub_key, "extraction_returned_empty_data")
                        logger.warning(f"  ⚠️  Chunk {idx+1}/{len(chunks)} of {filename} returned empty extraction data.")
                        continue

                except Exception as e:
                    self.metrics["chunks_failed"] += 1
                    self.metrics["llm_failures"] += 1
                    logger.error(f"  ⚠️  Chunk {idx+1}/{len(chunks)} of {filename} failed: {e}")
                    self._add_to_retry_queue("memory", sub_key, f"Exception: {str(e)[:50]}")
                    continue

                for c in data.get("characters", []):
                    if not isinstance(c, dict):
                        continue
                    raw_name = c.get("canonical_name") or c.get("name", "")
                    if not raw_name or str(raw_name).strip().lower() in ["", "unknown", "none"]:
                        logger.warning(f"  ⚠️  Skipping character with invalid name: {raw_name}")
                        continue
                        
                    name = _canonicalize_name(str(raw_name))
                    cid = str(uuid.uuid4())[:8]
                    
                    visual_dna = {
                        "age": c.get("age", ""),
                        "gender": c.get("gender", ""),
                        "hair": c.get("hair", ""),
                        "eyes": c.get("eyes", ""),
                        "face": c.get("face", ""),
                        "build": c.get("build", ""),
                        "clothing": c.get("clothing", ""),
                        "accessories": c.get("accessories", ""),
                        "distinctive_features": c.get("distinctive_features", []),
                        "appearance_confidence": c.get("appearance_confidence", 0.0),
                        "role": c.get("role", "")
                    }
                    
                    current_state = c.get("current_state", {})
                    current_state["importance"] = c.get("importance", 5)

                    self.memory_db.add_character(cid, name,
                                                 visual_dna,
                                                 current_state)
                                                 
                for loc in data.get("locations", []):
                    if not isinstance(loc, dict):
                        continue
                    raw_name = loc.get("canonical_name") or loc.get("name", "")
                    if not raw_name or str(raw_name).strip().lower() in ["", "unknown", "none"]:
                        logger.warning(f"  ⚠️  Skipping location with invalid name: {raw_name}")
                        continue

                    name = _canonicalize_name(str(raw_name))

                    v_tags = loc.get("visual_tags", "")
                    if isinstance(v_tags, list):
                        v_tags = ", ".join(v_tags)
                        
                    self.memory_db.add_location(
                        name,
                        loc.get("description", ""),
                        v_tags,
                    )
                    
                for concept in data.get("world_concepts", []):
                    if not isinstance(concept, dict): continue
                    self.memory_db.add_world_concept(
                        concept.get("concept_type", "misc"),
                        concept.get("name", "Unknown"),
                        concept.get("description", ""),
                    )
                    
                for rel in data.get("relationships", []):
                    if not isinstance(rel, dict): continue
                    self.memory_db.add_relationship(
                        rel.get("char1", ""), rel.get("char2", ""),
                        rel.get("type", "other"), rel.get("description", ""),
                    )
                    
                for event in data.get("events", []):
                    if not isinstance(event, dict):
                        logger.warning(f"  ⚠️  Invalid event format: {event}")
                        continue
                    
                    raw_chars = event.get("involved_characters") or event.get("characters", [])
                    extracted_names = []
                    
                    if isinstance(raw_chars, str):
                        extracted_names = [n.strip() for n in raw_chars.split(",")]
                    elif isinstance(raw_chars, list):
                        for item in raw_chars:
                            if isinstance(item, str):
                                extracted_names.append(item.strip())
                            elif isinstance(item, dict):
                                name = item.get("name") or item.get("canonical_name", "")
                                if name and isinstance(name, str):
                                    extracted_names.append(name.strip())
                                    
                    extracted_names = [n for n in extracted_names if n]
                    inv_chars_str = ", ".join(extracted_names)

                    self.memory_db.add_event(
                        summary=event.get("summary", ""),
                        importance=event.get("importance", 5),
                        involved_characters=inv_chars_str,
                        location=event.get("location", ""),
                        source_chunk=sub_key
                    )

                # Save world style from first chunk only if not already saved
                style_file = os.path.join(self.pm.dirs["memory"], "world_style.txt")
                if not world_style_saved and not os.path.exists(style_file):
                    style = data.get("world_style", "")
                    if style:
                        with open(style_file, "w", encoding="utf-8") as f:
                            f.write(style)
                        world_style_saved = True
                        logger.info(f"  World style: {style[:80]}…")
                    elif getattr(self.extractor_llm, "last_call_was_fallback", False):
                        # Don't lock in a generic mock style for the whole project —
                        # retry this on the next run once the LLM is actually up.
                        logger.error(
                            "  ⚠️  World style extraction hit LLM fallback — "
                            "not saving, will retry on next run."
                        )
                elif os.path.exists(style_file):
                    world_style_saved = True # Already exists, don't re-extract
                    if idx == 0:
                        with open(style_file, "r", encoding="utf-8") as f:
                            style = f.read()
                        logger.info(f"  Loaded existing world style: {style[:80]}…")

                self.pm.save_checkpoint("memory", "done", sub_key=sub_key)

        chars = self.memory_db.get_all_characters()
        locs = self.memory_db.get_all_locations()
        logger.info(f"✅ Memory: {len(chars)} characters, {len(locs)} locations extracted")
        if getattr(self.extractor_llm, "fallback_count", 0) > 0:
            logger.warning(
                f"⚠️  Stage 2 had {self.extractor_llm.fallback_count} LLM fallback(s) out of "
                f"{self.extractor_llm.total_calls} calls — those chunks were skipped and will "
                f"retry automatically on the next run."
            )

    # ── Stage 3: Character Sheets ─────────────────────────────────────────────
    def stage_character_sheets(self):
        logger.info("─── Stage 3: Character Sheets ───")
        if not self.config.get("character_sheet.enabled", True):
            logger.info("  Character sheets disabled in config")
            return

        chars_dir = os.path.join(self.pm.dirs["memory"], "characters")
        os.makedirs(chars_dir, exist_ok=True)
        characters = self.memory_db.get_all_characters()

        if not characters:
            logger.warning("  No characters in memory — skipping character sheets")
            return

        world_style = ""
        style_file = os.path.join(self.pm.dirs["memory"], "world_style.txt")
        if os.path.exists(style_file):
            with open(style_file, "r", encoding="utf-8") as f:
                world_style = f.read().strip()

        # FIX: was `self.pm.config` — ProjectManager has no `.config` attribute
        # at all (only ConfigManager, `self.config`, does). This was a
        # guaranteed AttributeError on every run that reached this line,
        # i.e. every run with any character in memory.
        gen_all = self.config.get("project.generate_all_character_sheets", False)
        min_importance = self.config.get("project.min_character_importance", 7)

        for char in characters:
            importance = char.get("current_state", {}).get("importance", 5)
            if not gen_all and importance < min_importance:
                logger.info(f"  Skipping {char.get('canonical_name')}: importance {importance} < {min_importance}")
                continue

            char_id = char["id"]
            char_name = char["canonical_name"]
            dna = char.get("visual_dna", {})
            # FIX: same bug as prompter.py had — blanket str(v) over every
            # field dumped list values as literal Python repr text
            # ("['innocent expression', ...]") and metadata fields
            # (appearance_confidence, role) as if they were visual tags.
            # Exclude non-visual keys and flatten list fields properly.
            NON_VISUAL_DNA_KEYS = {"appearance_confidence", "role"}
            dna_tags = []
            for k, v in dna.items():
                if k in NON_VISUAL_DNA_KEYS or not v:
                    continue
                if isinstance(v, (list, tuple, set)):
                    dna_tags.extend(
                        str(item).strip() for item in v
                        if str(item).strip().lower() not in {"none", "unknown", "not specified"}
                    )
                else:
                    s = str(v).strip()
                    if s.lower() not in {"none", "unknown", "not specified"}:
                        dna_tags.append(s)
            dna_str = ", ".join(dna_tags)

            out_path = os.path.join(chars_dir, f"{char_id}.png")
            if os.path.exists(out_path):
                logger.info(f"  Sheet exists: {char_name} — skipping")
                continue

            logger.info(f"  Generating character sheet: {char_name}")
            self.image_gen.generate_character_sheet(
                char_id=char_id,
                char_name=char_name,
                dna_str=dna_str,
                output_dir=chars_dir,
                world_style=world_style,
            )

        logger.info("✅ Character sheets complete")

    # ── Stage 4: Visual Planning ──────────────────────────────────────────────
    def stage_visual_planning(self):
        logger.info("─── Stage 4: Visual Planning ───")
        if not getattr(self.planner_llm, "is_available", True):
            logger.error("❌ Stage 4 Aborted: No LLM provider reachable.")
            return

        from core.visual.planner import StoryboardPlanner
        from core.visual.prompter import PromptGenerator

        translated_files = sorted(
            f for f in os.listdir(self.pm.dirs["output"])
            if f.startswith("translated_")
        )
        if not translated_files:
            logger.warning("No translated files — run stage_translate first")
            return

        storyboard_path = os.path.join(self.pm.dirs["output"], "storyboard.json")
        
        all_panels = []
        if os.path.exists(storyboard_path):
            try:
                import json
                with open(storyboard_path, "r", encoding="utf-8") as f:
                    all_panels = json.load(f)
                logger.info(f"  Loaded {len(all_panels)} existing panels.")
            except Exception as e:
                logger.error(f"Failed to load storyboard.json: {e}")

        current_chapter = 1
        seq = 1
        if all_panels:
            max_chapter = max((p.get("chapter", 1) for p in all_panels), default=1)
            current_chapter = max_chapter + 1
            seq = max((p.get("sequence", 0) for p in all_panels), default=0) + 1

        planner = StoryboardPlanner(self.planner_llm, config=self.config.config)
        prompter = PromptGenerator(self.memory_db, config=self.config.config, llm_adapter=self.planner_llm)

        new_panels_added = False
        
        def create_narrative_blocks(text, min_w=800, max_w=1200):
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            blocks = []
            current_block = []
            current_words = 0
            
            for p in paragraphs:
                words = len(p.split())
                if current_words + words > max_w and current_words >= min_w:
                    blocks.append("\n\n".join(current_block))
                    current_block = [p]
                    current_words = words
                else:
                    current_block.append(p)
                    current_words += words
                    
            if current_block:
                blocks.append("\n\n".join(current_block))
            return blocks

        for filename in translated_files:
            if self.pm.is_complete("visual_planning", sub_key=filename):
                logger.info(f"  Visual planning for {filename} cached — skipping")
                continue

            # State object that carries over across chunks within the SAME chapter
            planner_state = {
                "current_location": "unknown",
                "time_of_day": "unknown",
                "weather": "unknown",
                "active_characters": []
            }

            file_path = os.path.join(self.pm.dirs["output"], filename)
            text = self.pm.read_input(file_path)
            blocks = create_narrative_blocks(text)

            for i, chunk_text in enumerate(blocks):
                chunk_key = f"{filename}_{i}"
                if self.pm.is_complete("visual_planning_chunk", sub_key=chunk_key):
                    continue

                logger.info(f"  Planning narrative block {i+1}/{len(blocks)} from {filename} (~ch{current_chapter})")
                
                try:
                    import time
                    if getattr(self.planner_llm, "is_cloud", False):
                        time.sleep(2)
                        
                    result = planner.plan_panels(chunk_text, current_state=planner_state, chapter=current_chapter, start_sequence=seq)
                    planner_state = result.get("state", planner_state)
                    panels = result.get("panels", [])
                    
                    if not panels:
                        logger.warning(f"  ⚠️ Block {i+1} yielded 0 panels.")
                        continue

                    # IMAGE BUDGET ENFORCEMENT
                    target_images = self.config.config.get("storyboard", {}).get("target_images_per_1000_words", 6)
                    chunk_words = len(chunk_text.split())
                    budget = max(1, round((chunk_words / 1000) * target_images))
                    
                    unmerged_panels = [p for p in panels if not p.get("merge_with_previous", False)]
                    
                    if len(unmerged_panels) > budget:
                        scored_panels = []
                        for p in unmerged_panels:
                            bt = p.get("beat_type", "action")
                            imp = p.get("importance", 5)
                            
                            is_protected = (bt in ["reveal", "combat"]) or (imp >= 8)
                            
                            # Lower score = dropped/merged first
                            score = imp
                            if imp <= 4: score -= 20
                            elif bt == "transition": score -= 15
                            elif bt == "reaction": score -= 10
                            elif bt == "emotion": score -= 8
                            elif bt == "dialogue": score -= 5
                            elif bt == "environment": score += 1
                            elif bt == "action": score += 2
                            elif bt == "object_focus": score += 2
                            
                            scored_panels.append({
                                "panel": p,
                                "score": score,
                                "protected": is_protected
                            })
                            
                        scored_panels.sort(key=lambda x: x["score"])
                        
                        over_budget_by = len(unmerged_panels) - budget
                        merged_count = 0
                        
                        for sp in scored_panels:
                            if merged_count >= over_budget_by:
                                break
                            if not sp["protected"]:
                                sp["panel"]["merge_with_previous"] = True
                                merged_count += 1
                                logger.info(f"    - Over budget: Merging {sp['panel']['id']} ({sp['panel']['beat_type']}, imp {sp['panel'].get('importance', 5)}, score {sp['score']})")

                    # Generate prompts for the panels
                    for p in panels:
                        p["chapter"] = current_chapter
                        prompt_data = prompter.generate_prompt_for_panel(p, chapter_number=current_chapter)
                        p["prompt"] = prompt_data["prompt"]
                        p["negative_prompt"] = prompt_data["negative_prompt"]
                        p["reference_images"] = prompt_data["reference_images"]
                        p["prompt_cache_key"] = prompt_data.get("cache_key", "")
                        p["generation_params"] = prompt_data.get("generation_params", {})
                        
                    all_panels.extend(panels)
                    seq += len(panels)
                    new_panels_added = True
                    self.pm.save_checkpoint("visual_planning_chunk", "done", sub_key=chunk_key)
                    logger.info(f"    + {len(panels)} panels (total: {len(all_panels)})")
                except Exception as e:
                    logger.error(f"  ⚠️  Block {i+1} failed: {e}")
                    continue

            self.pm.save_checkpoint("visual_planning", "done", sub_key=filename)

        if new_panels_added:
            import json
            self.pm.save_output("storyboard.json", json.dumps(all_panels, indent=2))
            self.pm.save_checkpoint("visual_planning", "done")
            logger.info(f"✅ Visual planning complete: {len(all_panels)} panels")
        else:
            logger.info("  No new panels planned.")


    # ── Stage 5: Image Generation ─────────────────────────────────────────────
    def stage_generation(self):
        logger.info("─── Stage 5: Image Generation ───")
        
        storyboard_path = os.path.join(self.pm.dirs["output"], "storyboard.json")
        if not os.path.exists(storyboard_path):
            logger.warning("storyboard.json not found — run stage_visual_planning first")
            return

        images_dir = os.path.join(self.pm.dirs["output"], "images")
        os.makedirs(images_dir, exist_ok=True)

        import json
        with open(storyboard_path, "r", encoding="utf-8") as f:
            all_panels = json.load(f)

        panels_to_process = []
        generated_count = 0

        for panel in all_panels:
            pid = panel["id"]
            # Skip if merged with previous
            if panel.get("merge_with_previous", False):
                continue
                
            output_path = os.path.join(images_dir, f"{pid}.png")
            p_hash = panel.get("prompt_cache_key", "")

            if os.path.exists(output_path) and self.pm.get_checkpoint_value("img_cache", pid) == p_hash:
                generated_count += 1
            else:
                panels_to_process.append(panel)

        if not panels_to_process:
            logger.info("  No new images to generate. Image generation already complete.")
            return

        logger.info(f"  Generating {len(panels_to_process)} new images…")

        for panel in panels_to_process:
            pid = panel["id"]
            output_path = os.path.join(images_dir, f"{pid}.png")
            p_hash = panel.get("prompt_cache_key", "")

            self.image_gen.generate_image(
                prompt=panel["prompt"],
                output_path=output_path,
                negative_prompt=panel.get("negative_prompt", ""),
                reference_image_paths=panel.get("reference_images", []),
                generation_params=panel.get("generation_params", {})
            )

            self.pm.save_checkpoint("img_cache", p_hash, sub_key=pid)
            generated_count += 1
            if generated_count % 10 == 0:
                logger.info(f"  Progress: {generated_count} images")

        logger.info(f"✅ Image generation complete: {generated_count} total generated")

    # ── Stage 6: Audio Generation ─────────────────────────────────────────────
    def stage_audio(self):
        logger.info("─── Stage 6: Audio (TTS) ───")
        
        storyboard_path = os.path.join(self.pm.dirs["output"], "storyboard.json")
        if not os.path.exists(storyboard_path):
            logger.warning("storyboard.json not found — run stage_visual_planning first")
            return

        audio_dir = os.path.join(self.pm.dirs["output"], "audio")
        os.makedirs(audio_dir, exist_ok=True)

        import json
        with open(storyboard_path, "r", encoding="utf-8") as f:
            panels = json.load(f)

        audio_generated_count = 0
        panels_to_process = []

        for panel in panels:
            pid = panel["id"]
            out_path = os.path.join(audio_dir, f"{pid}.wav")
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
                panels_to_process.append(panel)
            else:
                audio_generated_count += 1

        if not panels_to_process:
            logger.info("  No new audio to generate. Audio generation already complete.")
            return

        logger.info(f"  Generating {len(panels_to_process)} new audio files…")

        for panel in panels_to_process:
            pid = panel["id"]
            out_path = os.path.join(audio_dir, f"{pid}.wav")
            narration = panel.get("description", "").strip() or "..."
            self.audio_gen.generate_audio(narration, out_path)
            audio_generated_count += 1
            if audio_generated_count % 10 == 0:
                logger.info(f"  Progress: {audio_generated_count} audio files")

        logger.info(f"✅ Audio generation complete: {audio_generated_count} audio files")

    # ── Stage 7: Video Assembly ───────────────────────────────────────────────
    def stage_video(self):
        logger.info("─── Stage 7: Video Assembly ───")
        self._unload_image_gen()  # Free VRAM before video rendering

        from core.video.renderer import VideoRenderer
        renderer = VideoRenderer(self.pm.project_dir, config=self.config.config)
        renderer.render()
        logger.info("✅ Video assembly complete")

    # ── Stage 8: Export ───────────────────────────────────────────────────────
    def stage_export(self):
        logger.info("─── Stage 8: Export ───")
        final = os.path.join(self.pm.project_dir, "output", "videos", "final_video.mp4")
        if not os.path.exists(final):
            logger.warning("Final video not found — check video stage logs")
            return

        size_mb = os.path.getsize(final) / (1024 * 1024)
        logger.info(f"✅ Final video ready: {final} ({size_mb:.1f} MB)")

        # YouTube SEO Packaging
        logger.info("  Generating YouTube SEO metadata…")
        
        # Gather story context for better SEO (prevents hallucination)
        chars = [c["canonical_name"] for c in self.memory_db.get_all_characters()][:10]
        locs = [l["canonical_name"] for l in self.memory_db.get_all_locations()][:5]
        context = f"Characters: {', '.join(chars)}. Locations: {', '.join(locs)}."

        system = (
            "You are a YouTube SEO expert. Generate metadata for a manhwa/novel recap video. "
            "Output ONLY valid JSON with 'title', 'description', 'tags' (list), and 'chapters' (text)."
        )
        prompt = (
            f"Project: {self.project_name}. {context} "
            f"Generate a catchy, high-CTR title and a detailed description that "
            f"includes the story beats and characters mentioned."
        )
        
        seo_data_raw = self.planner_llm.generate_json(prompt, system_prompt=system)
        package_path = os.path.join(self.pm.dirs["export"], "upload_package.json")
        try:
            seo = json.loads(seo_data_raw)
            export_dir = self.pm.dirs["export"]
            
            with open(os.path.join(export_dir, "youtube_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(seo, f, indent=2)
            
            # Create a simple upload package manifest
            package = {
                "project": self.project_name,
                "video_file": os.path.abspath(final),
                "srt_file": os.path.abspath(os.path.join(self.pm.dirs["output"], "videos", "subtitles.srt")),
                "metadata": seo,
                "status": "ready_for_upload"
            }
            with open(package_path, "w", encoding="utf-8") as f:
                json.dump(package, f, indent=2)
                
            logger.info(f"✓ YouTube package ready in {export_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate SEO metadata: {e}")

        # Google Drive Upload
        if self.config.get("gdrive.enabled", False):
            logger.info("  Uploading results to Google Drive…")
            from core.publishing.drive_uploader import DriveUploader
            uploader = DriveUploader(config=self.config.config)
            
            # Upload final video
            uploader.upload_file(final)
            
            # Upload package
            if os.path.exists(package_path):
                uploader.upload_file(package_path)

        print(f"\n🎉 DONE! Final video: {final} ({size_mb:.1f} MB)\n")
