import sys
import os

from core.visual.planner import StoryboardPlanner
from models.llm_adapter import SmartLLMAdapter

def main():
    llm = SmartLLMAdapter()
    planner = StoryboardPlanner(llm)

    text = """Arthur walked through the forest.

The wind blew through the trees.

He remembered his father's words.

A branch snapped behind him.

Arthur turned around.

A wolf emerged from the darkness."""

    state = {
        "current_location": "dark forest",
        "time_of_day": "night",
        "weather": "windy",
        "active_characters": ["Arthur"],
        "character_states": {
            "Arthur": {
                "outfit": "leather tunic",
                "emotion": "cautious",
                "position": "walking"
            }
        }
    }

    print("Running forest test with Groq/Ollama/DeepSeek/Gemini routing...")
    result = planner.plan_panels(text, current_state=state)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
