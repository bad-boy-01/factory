import os
import json
from core.orchestrator import UnifiedPipeline

class MockLLMAdapter:
    def __init__(self):
        self.is_available = True
        self.is_cloud = False
        self.fallback_count = 0
        self.total_calls = 0

    def generate(self, *args, **kwargs):
        return "mock"
        
    def generate_json(self, *args, **kwargs):
        # Force a malformed JSON error
        return '{"this_is_not_valid_json'

    def unload_model(self):
        pass

def main():
    p = UnifiedPipeline("test_benchmark")
    p._llm = MockLLMAdapter()
    
    # Create dummy translated file
    os.makedirs(p.pm.dirs["output"], exist_ok=True)
    with open(os.path.join(p.pm.dirs["output"], "translated_ch1.txt"), "w") as f:
        f.write("Arthur woke up. He went to the mystic mountain to train.")
        
    print("Testing Stage Memory with failing LLM (malformed JSON)...")
    p.stage_memory()
    
    # Now simulate LLM Exhausted
    class ExhaustedAdapter(MockLLMAdapter):
        def generate_json(self, *args, **kwargs):
            from models.llm_adapter import LLMFallbackExhausted
            raise LLMFallbackExhausted("Simulated LLM exhaustion")
            
    p._llm = ExhaustedAdapter()
    print("\nTesting Stage Visual Planning with LLM Exhausted...")
    p.stage_visual_planning()
    
    # Save metrics (normally done at end of run_all)
    p._save_metrics()
    
    print("\n--- Verifying run_metrics.json ---")
    metrics_path = os.path.join(p.pm.project_dir, "artifacts", "run_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            print(json.dumps(json.load(f), indent=2))
    
    print("\n--- Verifying retry_queue.json ---")
    queue_path = os.path.join(p.pm.dirs["output"], "retry_queue.json")
    if os.path.exists(queue_path):
        with open(queue_path) as f:
            print(json.dumps(json.load(f), indent=2))
            
    print("\n--- Verifying raw_llm directory ---")
    raw_dir = os.path.join(p.pm.dirs["output"], "raw_llm")
    if os.path.exists(raw_dir):
        files = os.listdir(raw_dir)
        print("Files in raw_llm:", files)
        if files:
            with open(os.path.join(raw_dir, files[0])) as f:
                print(f"Sample content ({files[0]}):", f.read()[:50])

if __name__ == "__main__":
    main()
