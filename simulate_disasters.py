import os
import json
import logging
import shutil
import sys
import time
from unittest.mock import patch, PropertyMock

# Suppress verbose logs to keep output clean
logging.getLogger("core").setLevel(logging.WARNING)
logging.getLogger("NVF").setLevel(logging.WARNING)

PROJ_NAME = "disaster_test"
BASE_DIR = os.getcwd()
PROJ_DIR = os.path.join(BASE_DIR, "projects", PROJ_NAME)

from core.orchestrator import UnifiedPipeline
from core.visual.planner import ScenePlanner

# NOTE: these tests exercise checkpoint/crash-resume logic in
# orchestrator.py, not real LLM connectivity — they previously worked by
# accident, because Groq/Ollama would fail in any sandboxed/offline test
# environment and the orchestrator used to accept the resulting mock
# content as if it were valid scenes. That's exactly the bug fixed in
# models/llm_adapter.py / core/orchestrator.py (see
# FALLBACK_FAILURE_ANALYSIS.md): mock content is now correctly rejected
# rather than silently used, so these tests need an explicit, deterministic
# stand-in for a *working* LLM to keep testing what they're actually meant
# to test. This patches ScenePlanner.plan_scenes directly — one level above
# the LLM call — so no network access is needed and the fake output is
# never mistaken for fallback content (no `_mock_fallback` key).
def _fake_plan_scenes(self, text_chunk, chapter=1, events=None):
    return [
        {
            "scene_id": "SC001",
            "location": "Test Location",
            "characters": ["Test Character"],
            "emotion": "neutral",
            "action": "a scene happens",
            "camera_angle": "medium shot",
            "lighting": "natural daylight",
            "visual_prompt_tags": "1person, interior scene",
            "narration_text": text_chunk[:200],
            "complexity": 5,
            "chapter": chapter,
        },
        {
            "scene_id": "SC002",
            "location": "Test Location",
            "characters": ["Test Character"],
            "emotion": "neutral",
            "action": "the scene continues",
            "camera_angle": "wide shot",
            "lighting": "natural daylight",
            "visual_prompt_tags": "1person, interior scene",
            "narration_text": text_chunk[200:400] or text_chunk[:200],
            "complexity": 5,
            "chapter": chapter,
        },
    ]

_plan_scenes_patch = patch.object(ScenePlanner, "plan_scenes", _fake_plan_scenes)

def reset_project(num_files=2):
    import gc
    gc.collect()
    time.sleep(0.5)
    if os.path.exists(PROJ_DIR):
        try:
            shutil.rmtree(PROJ_DIR)
        except PermissionError:
            time.sleep(1)
            shutil.rmtree(PROJ_DIR)
    os.makedirs(os.path.join(PROJ_DIR, "input"))
    os.makedirs(os.path.join(PROJ_DIR, "output"))
    os.makedirs(os.path.join(PROJ_DIR, "memory"))
    for i in range(1, num_files + 1):
        with open(os.path.join(PROJ_DIR, "output", f"translated_file{i}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Chapter {i}. This is the text for chapter {i}. " * 10)

def test_1():
    print("\n=== TEST 1: Crash after clips.json save, before checkpoint ===")
    reset_project()
    with _plan_scenes_patch, patch("models.llm_adapter.SmartLLMAdapter.is_available", new_callable=PropertyMock, return_value=True):
        pipeline = UnifiedPipeline(PROJ_NAME)

        original_save_checkpoint = pipeline.pm.save_checkpoint
        crash_flag = {"crashed": False}

        def buggy_save_checkpoint(stage, value, sub_key=""):
            if stage == "visual_planning" and sub_key == "translated_file1.txt" and not crash_flag["crashed"]:
                crash_flag["crashed"] = True
                print("  [Simulated Crash] Kaggle kernel died just before saving checkpoint for file1")
                raise RuntimeError("Kernel Died")
            original_save_checkpoint(stage, value, sub_key)

        with patch.object(pipeline.pm, 'save_checkpoint', side_effect=buggy_save_checkpoint):
            try:
                pipeline.stage_visual_planning()
            except RuntimeError:
                pass

        clips_path = os.path.join(PROJ_DIR, "output", "clips.json")
        with open(clips_path, "r") as f:
            clips = json.load(f)
        scenes_before = [s for c in clips for s in c.get("shots", [])]
        print(f"  Scenes saved to disk before crash: {len(scenes_before)}")

        print("  [Resuming Pipeline]")
        pipeline = UnifiedPipeline(PROJ_NAME)
        pipeline.stage_visual_planning()

        with open(clips_path, "r") as f:
            clips = json.load(f)
        scenes_after = [s for c in clips for s in c.get("shots", [])]
        chapters = [s.get("chapter") for s in scenes_after]

        print(f"  Scenes after resume: {len(scenes_after)}")
        print(f"  Chapters present: {list(set(chapters))}")

        ch1_count = chapters.count(1)
        ch2_count = chapters.count(2)
        print(f"  Ch1 scenes: {ch1_count}, Ch2 scenes: {ch2_count}")
        assert len(scenes_before) > 0, "Scenes lost!"
        assert ch1_count > 0 and ch1_count == len(scenes_before), "Duplication detected!"
        print("  ✅ TEST 1 PASSED: No scene loss, no duplication")

def test_2():
    print("\n=== TEST 2: Crash during second file processing ===")
    reset_project()
    with _plan_scenes_patch, patch("models.llm_adapter.SmartLLMAdapter.is_available", new_callable=PropertyMock, return_value=True):
        pipeline = UnifiedPipeline(PROJ_NAME)

        original_chunk_text = sys.modules['core.orchestrator']._chunk_text

        def buggy_chunk_text(text, *args, **kwargs):
            if "Chapter 2" in text:
                print("  [Simulated Crash] Kernel died while processing file2.txt")
                raise RuntimeError("Kernel Died")
            return original_chunk_text(text, *args, **kwargs)

        with patch('core.orchestrator._chunk_text', side_effect=buggy_chunk_text):
            try:
                pipeline.stage_visual_planning()
            except RuntimeError:
                pass

        file1_done = pipeline.pm.is_complete('visual_planning', 'translated_file1.txt')
        print(f"  File1 checkpoint status: {'Complete' if file1_done else 'Incomplete'}")

        print("  [Resuming Pipeline]")
        pipeline = UnifiedPipeline(PROJ_NAME)
        pipeline.stage_visual_planning()

        clips_path = os.path.join(PROJ_DIR, "output", "clips.json")
        with open(clips_path, "r") as f:
            clips = json.load(f)
        chapters = [s.get("chapter") for c in clips for s in c.get("shots", [])]

        print(f"  Chapters in final clips.json: {list(set(chapters))}")
        assert 1 in chapters and 2 in chapters, "File skipping failed!"
        print("  ✅ TEST 2 PASSED: File 1 skipped, File 2 recovered successfully")

def test_3():
    print("\n=== TEST 3: 0-byte audio file resume ===")
    reset_project(1)
    pipeline = UnifiedPipeline(PROJ_NAME)
    
    clips_path = os.path.join(PROJ_DIR, "output", "clips.json")
    fake_clips = [{"shots": [{"scene_id": "SC001", "narration_text": "Test audio generation resume"}]}]
    with open(clips_path, "w") as f:
        json.dump(fake_clips, f)
        
    audio_dir = os.path.join(PROJ_DIR, "output", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    wav_path = os.path.join(audio_dir, "SC001.wav")
    with open(wav_path, "wb") as f:
        pass # Create 0 byte file
        
    print(f"  Created corrupted audio file: {wav_path} (size: {os.path.getsize(wav_path)} bytes)")
    
    # Run audio stage
    pipeline.stage_audio()
    
    size = os.path.getsize(wav_path)
    print(f"  File size after resume: {size} bytes")
    assert size > 100, "Audio was not regenerated!"
    print("  ✅ TEST 3 PASSED: Audio successfully regenerated")

def test_4():
    print("\n=== TEST 4: Interrupt checkpoint save ===")
    reset_project()
    pipeline = UnifiedPipeline(PROJ_NAME)
    
    pipeline.pm.save_checkpoint("test_key", "valid_data")
    chk_path = pipeline.pm.checkpoint_file
    
    with open(chk_path, "r") as f:
        data = json.load(f)
    print(f"  Original checkpoint data: {data}")
    
    original_replace = os.replace
    def buggy_replace(src, dst):
        print("  [Simulated Crash] Power failure during os.replace")
        raise OSError("I/O Error during atomic swap")
        
    with patch('os.replace', side_effect=buggy_replace):
        try:
            pipeline.pm.save_checkpoint("test_key_2", "invalid_data")
        except Exception:
            pass
            
    with open(chk_path, "r") as f:
        data_after = json.load(f)
        
    print(f"  Checkpoint data after crash: {data_after}")
    assert "test_key_2" not in data_after
    assert data_after.get("test_key") == "valid_data", "Original data corrupted!"
    print("  ✅ TEST 4 PASSED: Atomic logic prevented corruption")

def test_5():
    print("\n=== TEST 5: Large project scalability (1000+ scenes) ===")
    reset_project(0)
    
    # Create 50 files to force 50 incremental updates
    for i in range(1, 51):
        with open(os.path.join(PROJ_DIR, "output", f"translated_file{i}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Chapter {i}. " * 10)
            
    pipeline = UnifiedPipeline(PROJ_NAME)

    t0 = time.time()
    with _plan_scenes_patch, patch("models.llm_adapter.SmartLLMAdapter.is_available", new_callable=PropertyMock, return_value=True):
        pipeline.stage_visual_planning()
    t1 = time.time()
    
    clips_path = os.path.join(PROJ_DIR, "output", "clips.json")
    with open(clips_path, "r") as f:
        clips = json.load(f)
        
    scenes = [s for c in clips for s in c.get("shots", [])]
    print(f"  Total generated scenes: {len(scenes)}")
    print(f"  Time taken for 50 incremental JSON updates: {t1 - t0:.2f} seconds")
    assert len(scenes) >= 50, "Scalability test failed to process files"
    assert (t1 - t0) < 15.0, "JSON rewrite is an O(n^2) bottleneck!"
    print("  ✅ TEST 5 PASSED: Incremental persistence remains performant")

if __name__ == "__main__":
    print("Starting Disaster Recovery Simulations...")
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    print("\n🏆 ALL SIMULATIONS PASSED SUCESSFULLY.")