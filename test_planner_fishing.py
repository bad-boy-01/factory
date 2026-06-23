import sys
import os

from core.visual.planner import StoryboardPlanner
from models.llm_adapter import SmartLLMAdapter

def main():
    llm = SmartLLMAdapter()

    planner = StoryboardPlanner(llm)

    text = """"Changshou is awake! Come quickly, Changshou is awake!"
"Little brother is awake!"
"Third brother! Third brother!"
Noisy shouts filled Xu Changshou's ears. When he opened his eyes, he saw many faces of various sizes looking at him joyfully under the dilapidated roof. Three male and two female gathered at the bedside, shouting incessantly. The smallest, dirty girl was shaking his body with all her strength.
Xu Changshou felt very weak. This little girl shook him until he was dizzy. Just as he was about to faint, a wave of memories that did not belong to him suddenly enveloped his brain.
The descriptions were incredibly clear, as if he had experienced them himself. Only then did he realize that he had truly undergone transmigration. In the past, he had graduated from a prestigious university, started his own business after graduation, experienced failures in business life, but later rose again. However, he was betrayed by the wife he loved most, and she put poison in his wine. Finally, he took his last breath and dragged that woman down from the high building with him.
He didn't expect the Heavens to give him another chance at life. The name of the body he was currently in was Xu Changshou, and he was only six years old. Because he had suffered from Qi disease since birth, he remained weak and listless.
The sun-burned man in his thirties across from him was named Xu Kaixi, who was the father of this body. The woman holding his hand next to him was his mother, Zhang Xiue. He had two older brothers: one was sixteen-year-old Xu Dagui, and the other was eleven-year-old Xu Youtian. Below him, he had a five-year-old little sister named Xu Xiaomei.
His brother Xu Dagui was already married; except for the absence of his sister-in-law Zheng Chunni, the old Xu family was now together in full force. Looking at the worried facial expressions of the five people in front of him, it was impossible for Xu Changshou not to be moved.
In his previous life, he was an orphan and the woman he loved with all his heart had finally betrayed him. Since he could live again and feel family love, he thought the situation was not that bad."""

    state = {
        "current_location": "unknown",
        "time_of_day": "unknown",
        "weather": "unknown",
        "active_characters": []
    }

    print("Running planner with Groq/Ollama/DeepSeek/Gemini routing...")
    result = planner.plan_panels(text, current_state=state)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
