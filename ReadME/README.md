# 🎬 Novel Video Factory v4.5 (NVF)

Convert any novel script into a **Korean manhwa-style video** — 100% FREE, runs on Kaggle T4 GPU.

## New in v4.5

- 🚀 **Long Prompt Support:** Integrated `Compel` to handle prompts >77 tokens without truncation.
- 🎭 **Face-Specialized Identity:** Switched to `ip-adapter-plus-face` and raised scale (0.65) for superior character consistency.
- 🛡️ **Mock Content Rejection:** Pipeline now detects and rejects fabricated LLM fallbacks (no more "Hero at Training Grounds" hallucinations).
- 💬 **Smart Captions:** Captions now split by line-width and time proportionally to word count.
- 📈 **SEO Context:** YouTube metadata generation now uses real character/location data from memory.
- ⚡ **Performance:** Conditional delays for cloud providers (Groq) only; local Ollama runs at full speed.

## What it does

1. 📖 **Reads** your novel (.txt, .md, .epub)
2. 🌐 **Translates** Korean/Chinese/any language → English (skips if already English)
3. 🧠 **Extracts** characters, locations, and lore into a memory database
4. 🎨 **Draws** each scene as a manhwa panel (Animagine-XL 3.1)
5. 🔊 **Narrates** with Microsoft Edge TTS (free, natural voices)
6. 🎥 **Assembles** 10-minute clips → stitches into final 2-hour video

## Quick Start (Kaggle)

1. Upload this project to Kaggle
2. Add your novel to `projects/novel/input/chapter1.txt`
3. (Optional) Add `GROQ_API_KEY` to Kaggle Secrets → [console.groq.com](https://console.groq.com) (free, no card)
4. Run `Kaggle_NVF_v4.ipynb` cell by cell

## Free Models Used

| Stage | Model | Cost |
|-------|-------|------|
| LLM | Groq llama-3.3-70b OR Ollama qwen2.5:7b | FREE |
| Images | cagliostrolab/animagine-xl-3.1 (SDXL) | FREE |
| Audio | Microsoft Edge TTS (edge-tts) | FREE |
| Video | MoviePy + FFmpeg | FREE |

## Project Structure

```
NVF_v4.5/
├── config/
│   └── default.yaml          ← All settings here
├── core/
│   ├── orchestrator.py       ← Main pipeline coordinator
│   ├── translation/          ← Smart language detection + translation
│   ├── memory/               ← Character/location knowledge store
│   ├── visual/               ← Scene planning + prompt generation
│   ├── video/                ← Video assembly + Ken Burns effect
│   └── publishing/           ← SEO metadata generator
├── models/
│   ├── llm_adapter.py        ← Groq + Ollama adapters
│   ├── image_adapter.py      ← Animagine-XL 3.1 (manhwa style)
│   └── audio_adapter.py      ← Edge TTS narration
├── projects/
│   └── novel/
│       └── input/            ← Put your .txt novel files here
├── main.py                   ## CLI entry point
├── Kaggle_NVF_v4.ipynb       ## Kaggle notebook
├── kaggle_setup.sh           ## One-time setup script
└── requirements.txt
```

## CLI Usage

```bash
# Run full pipeline
python main.py novel

# Import and run
python main.py novel --input /path/to/novel.txt

# Resume from a specific stage (safe after disconnect)
python main.py novel --stage generation
python main.py novel --stage audio
python main.py novel --stage video

# All stage options:
# translate | memory | char_sheets | visual | generation | audio | video | export
```

## Config (`config/default.yaml`)

Key settings to change:

```yaml
project:
  language:
    source: "auto"     # "auto" detects language, or set "Korean" / "Chinese" / "English"

models:
  llm:
    provider: "groq"   # "groq" (free cloud) or "ollama" (local)
  image:
    width: 832         # 832×480 = fast; 1344×768 = better quality (4× slower)
    height: 480

audio:
  voice: "en-US-AndrewNeural"  # Change for different narrator voice
```

## Resuming After Kaggle Disconnect

Every stage saves checkpoints. To resume:
- Re-run Cell 1 (setup) to restart Ollama
- Re-run Cell 2 (config)
- Jump to the cell for the stage where you left off
- Already-completed work is automatically skipped

## Bug Fixes vs v4

- ✅ **CLIP Truncation:** Compel integration ensures "korean manhwa style" tags are never dropped.
- ✅ **Hallucination Prevention:** Mock LLM responses are rejected in Stage 2/4.
- ✅ **Character Consistency:** Face-plus IP-Adapter weight + higher scale (0.65).
- ✅ **Caption Splitting:** Proportional timing and line-width-based grouping.
- ✅ **Ollama Speed:** Removed unnecessary delays when using local providers.
- ✅ **SEO Context:** Metadata prompt now includes real character/location data.

## Character Consistency

Each character gets 6 reference pose images (front, smile, angry, crying, fight, sit).
These are fed to IP-Adapter which "locks in" the character's appearance across all scenes.

v4.5 uses `ip-adapter-plus-face_sdxl_vit-h.bin` at scale `0.65` for optimal identity retention.

## 2-Hour Video Strategy

- Kaggle T4 can run ~10 minutes of video per session
- The pipeline automatically splits into `clip01.mp4`, `clip02.mp4`, etc.
- Run across multiple sessions; at the end `stage_video` stitches everything with FFmpeg
- No quality loss — FFmpeg stream-copies (no re-encode)

Current run commands - 
!git clone https://github.com/bad-boy-01/NFV_v4.5-master.git
!cd NFV_v4.5-master && bash kaggle_setup.sh
import subprocess
import time
import os
print("Starting Ollama server...")
# Start Ollama in the background properly
process = subprocess.Popen(
["ollama", "serve"],
stdout=subprocess.DEVNULL,
stderr=subprocess.DEVNULL
)

# Give it 5 seconds to boot up
time.sleep(5)
print("Pulling qwen2.5:7b model (this may take a minute if not cached)...")
# Run the pull command synchronously so we wait for it to finish
os.system("ollama pull qwen2.5:7b")
print("Ollama is ready!")
%cd NFV_v4.5-master
!python main.py novel --input projects/novel/input/chapter1.txt