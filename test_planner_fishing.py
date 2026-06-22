import sys
import os

from core.visual.planner import StoryboardPlanner
from models.llm_adapter import SmartLLMAdapter

def main():
    llm = SmartLLMAdapter()

    planner = StoryboardPlanner(llm)

    text = """I sat beside the river fishing.
    
I wondered what to cook tonight.

The float suddenly dipped.

A huge fish burst from the water."""

    state = {
        "current_location": "riverbank",
        "time_of_day": "sunset",
        "weather": "clear",
        "active_characters": ["Arthur"],
        "character_states": {
            "Arthur": {
                "outfit": "leather tunic",
                "emotion": "calm",
                "position": "sitting"
            }
        }
    }

    print("Running planner with Groq/Ollama/DeepSeek/Gemini routing...")
    result = planner.plan_panels(text, current_state=state)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
