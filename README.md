# Novel Video Factory v5 — Beat-Based Storyboard Architecture

Welcome to Novel Video Factory v5! The entire visual planning architecture has been overhauled from a word-count-based chunking system into a **Beat-Based Storyboard Architecture**. This massive upgrade forces the LLM to think like a manga or manhwa artist, resulting in highly cohesive storytelling, significantly reduced GPU rendering costs, and strictly enforced character continuity.

## The V5 Architecture Pipeline

The pipeline now executes sequentially to create a seamless storyboard video:
1. **Novel Text Extraction**: The raw text is translated and split into narrative blocks (800-1200 words).
2. **Memory Engine Extraction**: Automatically tracks and stores character `visual_dna`, `current_outfit`, and location metadata to the SQLite database.
3. **Storyboard Planning**: The `StoryboardPlanner` loops over narrative blocks and extracts *visual beats* (e.g., action, reaction, emotion, object_focus) rather than isolating sentence-by-sentence frames.
4. **Prompt Assembly**: The pipeline injects granular character DNA and outfit states directly into Animagine XL / Stable Diffusion prompts.
5. **Asset Generation**: High-importance beats (score 8-10) receive premium GPU budgets (40 steps, high CFG). Low-importance beats (score <= 2) trigger a new `merge_with_previous` flag.
6. **Video Rendering**: `VideoRenderer` consumes the `storyboard.json` directly. Panels marked with `merge_with_previous` reuse the previous image asset, chaining camera movements (Ken Burns effects) and subtitles without burning GPU cycles on redundant image generation.

## Using the Incremental Workflow

To begin working on a subsequent part of your novel, simply place your new script, such as `novel_name_part_2.txt`, into the `projects/<your_project_name>/input/` directory. You can initiate the pipeline using the standard command: 
```bash
python main.py <your_project_name>
```
The pipeline intelligently skips previously completed stages and focuses solely on generating the missing beats, seamlessly extending the `storyboard.json` and appending the final video.

## Core V5 Enhancements

| Stage | V5 Behavior |
| :--- | :--- |
| **Visual Planning** | The planner operates strictly on a `storyboard.json` schema, passing a `state` object (weather, time, location, character outfits) between blocks to eliminate continuity drift. |
| **Asset Generation** | Utilizes dynamic generation budgets based on LLM-assigned `importance` scores. |
| **Video Rendering** | Applies custom Ken Burns panning and zooming directly correlated to the `beat_type` (e.g. `reveal` triggers a slow zoom out) and `shot_type` (e.g. `close_up` triggers slower panning). |

## Essential Backup Checklist

To ensure you can successfully resume your project at any time, please verify that your backup includes the following essential components:

*   **Memory Directory**: Contains the SQLite database (`novel_memory.db`) and character reference sheets.
*   **Checkpoints File**: Tracks the completion status of every file and stage.
*   **Storyboard File**: The `storyboard.json` file in the output directory, which serves as the blueprint for the entire video.
*   **Asset Folders**: The `images` and `audio` directories containing all generated media.
