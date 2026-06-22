# NVF v5 — Progress & Architecture Guide

## Overview
This document serves as a persistent guide for the **Novel Video Factory (NVF) v5** architecture. It details the massive paradigm shift implemented to move from a rigid "word chunking" pipeline to a fluid, beat-based storyboard director.

---

## 1. Summary of Changes Made (v5 Migration)

Today's updates successfully upgraded the pipeline from v4/v4.5 to v5. The core focus was **reducing redundant image generation, cutting GPU costs, and enforcing strict narrative/visual continuity.**

### Architectural Overhauls
- **Beat-Based Storyboarding:** Removed the legacy `ScenePlanner` and chunk-based processing. Replaced it with the `StoryboardPlanner` which extracts chronological *visual beats* (e.g., `environment`, `emotion`, `reveal`, `action`) rather than splitting scenes by arbitrary sentence counts.
- **Narrative Blocks:** Replaced blind 1000-word text chunking with semantic paragraph-level grouping (800–1200 words) in the `orchestrator.py` to preserve narrative flow.
- **State Tracking:** Implemented a continuous `state` object (tracking `current_location`, `weather`, `time_of_day`, and granular `character_states` like outfits and positions) that is passed between LLM calls to prevent continuity drift and hallucinations (e.g., characters suddenly changing clothes).
- **Legacy Cleanup:** Deleted obsolete scripts like `coverage_validator.py`, `scene_validator.py`, and `continuity_validator.py`. 

### Budget & Cost Optimization
- **Dynamic Image Budgeting:** The `orchestrator.py` now calculates an image budget based on word count (targeting ~7 images per 1000 words).
- **Python-Enforced Merging:** If the LLM generates too many unmerged panels, the orchestrator applies a weighted `effective_score` to each panel (`importance + beat_weight`) and forces `merge_with_previous = True` on the lowest-scoring beats.
- **Flashback Prevention:** Instructed the LLM to represent internal memories and thoughts as `emotion` beats occurring in the *current* location, explicitly prohibiting the generation of visual flashbacks.
- **Calibrated Importance & Merge Thresholds:** Tweaked the LLM prompt to grade importance properly (1-10) and automatically merge `emotion` and `reaction` beats if importance is <= 4, avoiding expensive new generations for minor internal monologue.

---

## 2. How the Project Works (For Future Agents)

If you are an AI reading this, here is the exact mental model you need to understand NVF v5:

### The Pipeline Flow
1. **Extraction & Memory:** Raw novel text is split into Narrative Blocks. The memory engine parses characters, relationships, and global world context, saving it to `novel_memory.db`.
2. **Storyboard Planning (`core/visual/planner.py`):** 
   - The LLM receives the text block + the *current physical state* of the world.
   - It outputs a JSON array of `panels`.
   - The goal is **not** to generate an image for every sentence. The goal is to identify **Key Visual Moments**.
   - If a character has a thought or minor reaction, it becomes an `emotion` beat with a low importance score and `merge_with_previous: true`.
3. **Budget Enforcement (`core/orchestrator.py`):**
   - The python orchestrator acts as a safeguard against overly verbose LLMs.
   - It counts the number of generated images requested by the LLM. If the count exceeds the block's budget, it ranks the panels by importance and forces the least important ones to merge. 
   - **Protected Beats:** `reveal` and `combat` (or any beat with importance >= 9) are sacred and are never forced to merge.
4. **Prompt Assembly & Image Gen:** Generates high-quality prompts from the dense descriptions, injecting character visual DNA and current outfits.
5. **Video Rendering (`core/video/renderer.py`):**
   - Reads the final `storyboard.json`.
   - If a beat has `merge_with_previous: true`, the renderer **reuses the previous image asset**. It extends the timeline, applies a subtle Ken Burns zoom/pan effect based on the beat type, and layers the new subtitles. This creates a highly cinematic, dynamic storyboard flow while bypassing expensive Stable Diffusion calls.

### Development Guidelines
- **DO NOT** reintroduce scene-validation loops or word-count-based chunk retries.
- **DO NOT** let the LLM generate prompts directly. Prompts are assembled downstream.
- If an issue arises with too many images being generated, tune the `importance` logic in the prompt or adjust the python budget enforcement limits in `orchestrator.py`. Do not rely solely on prompting to limit generation counts; always use programmatic Python post-processing to cap budgets.
