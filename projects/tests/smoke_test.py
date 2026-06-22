import os
import sys
import time
import logging
import json
import torch
import psutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.orchestrator import UnifiedPipeline

# Configure logging to both file and console
log_file = os.path.join('projects', 'tests', 'smoke_test.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SmokeTest')

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    return 0

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def prewarm_ollama(model_name: str):
    """Avoids first-request timeouts by pre-loading the model."""
    logger.info(f"Pre-warming Ollama model: {model_name}...")
    try:
        # Ping model with a tiny request
        import requests
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "ping", "stream": False},
            timeout=120
        )
        logger.info(f"Model {model_name} is ready.")
    except Exception as e:
        logger.warning(f"Could not pre-warm {model_name}: {e}")

def run_smoke_test():
    project_name = f"SmokeTest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    source_file = os.path.join('projects', 'tests', 'smoke_story.txt')

    # Check if strict mode is requested (can be passed via env or config)
    from core.config_manager import ConfigManager
    config_obj = ConfigManager('config/default.yaml')
    strict_mode = config_obj.get('system.strict_mode', False)

    report = {
        "project": project_name,
        "start_time": datetime.now().isoformat(),
        "stages": [],
        "total_duration": 0,
        "status": "FAILED",
        "strict_mode": strict_mode,
        "mock_usage_detected": False
    }

    start_all = time.time()

    try:
        logger.info(f"Starting End-to-End Smoke Test: {project_name} (Strict: {strict_mode})")

        # Pre-warm models to avoid timeout on first run
        prewarm_ollama("qwen2.5:7b")
        prewarm_ollama("deepseek-r1:8b")

        pipeline = UnifiedPipeline(project_name)

        stages = [
            ("Import", lambda: pipeline.import_source(source_file)),
            ("Translate", pipeline.stage_translate),
            ("Memory", pipeline.stage_memory),
            ("Character Sheets", pipeline.stage_character_sheets),
            ("Visual Planning", pipeline.stage_visual_planning),
            ("Generation", pipeline.stage_generation),
            ("Audio", pipeline.stage_audio),
            ("Video", pipeline.stage_video),
            ("Export", pipeline.stage_export)
        ]

        for stage_name, stage_fn in stages:
            logger.info(f">>> Running Stage: {stage_name}")
            start_stage = time.time()
            vram_before = get_vram_usage()
            ram_before = get_ram_usage()

            try:
                stage_fn()
                status = "PASS"
            except Exception as e:
                logger.error(f"Stage {stage_name} FAILED: {e}", exc_info=True)
                status = "FAIL"
                raise e

            duration = time.time() - start_stage
            vram_after = get_vram_usage()
            ram_after = get_ram_usage()

            report["stages"].append({
                "name": stage_name,
                "status": status,
                "duration": round(duration, 2),
                "vram_mb": round(vram_after, 2),
                "ram_mb": round(ram_after, 2),
                "vram_delta": round(vram_after - vram_before, 2)
            })

        # Final Audit: Check for Mock poison in logs or outputs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
                if "Using schema-valid mock" in log_content or "[MOCK" in log_content:
                    report["mock_usage_detected"] = True
                    if strict_mode:
                        logger.error("MOCK USAGE DETECTED in strict mode. Failing test.")
                        report["status"] = "FAILED_DUE_TO_MOCKS"
                        return report

        report["status"] = "SUCCESS"
        logger.info("End-to-End Smoke Test COMPLETED SUCCESSFULLY.")

    except Exception as e:
        logger.error(f"Smoke Test ABORTED due to error: {e}")
        report["status"] = "FAILED"
        report["error"] = str(e)

    report["total_duration"] = round(time.time() - start_all, 2)
    report["end_time"] = datetime.now().isoformat()

    # Save report
    report_path = os.path.join('projects', 'tests', 'smoke_test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    logger.info(f"Smoke test report saved to {report_path}")

    return report

if __name__ == "__main__":
    run_smoke_test()
