# Novel Video Factory v4 — Part 2 Friendly Workflow

The Novel Video Factory has been enhanced to support a seamless, incremental workflow designed for multi-part projects. This update ensures that you can continue your story weeks or months later by simply adding new script files to your existing project structure. The system will automatically detect new content, preserve your established characters and world style, and generate only the necessary new assets to complete the next part of your video.

### Using the Incremental Workflow

To begin working on a subsequent part of your novel, first ensure you have a backup of your entire project folder. Place your new script, such as `novel_name_part_2.txt`, into the `projects/<your_project_name>/input/` directory. You can then initiate the pipeline using the standard command: `python main.py <your_project_name>`. Alternatively, you may explicitly import the new script by providing the file path: `python main.py <your_project_name> --input path/to/novel_name_part_2.txt`. The pipeline is now intelligent enough to skip previously completed stages and focus solely on the newly added material.

### Core Enhancements and Consistency

The updated architecture prioritizes consistency and efficiency across all stages of production. By leveraging the existing `novel_memory.db` and previously generated character sheets, the system ensures that Arthur, Merlin, and all other characters maintain their visual identity throughout the entire series. Similarly, the world style extracted during the initial run is preserved in `world_style.txt`, ensuring that the artistic atmosphere remains uniform.

| Stage | Incremental Behavior |
| :--- | :--- |
| **Translation** | Automatically skips files that have already been translated. |
| **Memory Extraction** | Processes only new text chunks to identify new entities while retaining existing knowledge. |
| **Visual Planning** | Loads the master `clips.json` and appends new scenes for subsequent chapters. |
| **Asset Generation** | Generates images and audio only for scenes that do not already have cached files. |
| **Video Rendering** | Uses content hashing to detect changes, re-rendering only the clips that have been updated. |

### Bug Fixes and Stability

Several critical issues have been addressed to improve the reliability of the factory. A major bug in the video renderer was resolved where the system incorrectly searched for `shot_id` instead of the standard `scene_id`. Additionally, logic within the `ClipBuilder` and `ScenePlanner` was refactored to provide stable, non-shifting scene IDs. This ensures that adding a "Part 2" will not disrupt the numbering or organization of "Part 1," making the project truly future-proof.

### Essential Backup Checklist

To ensure you can successfully resume your project at any time, please verify that your backup includes the following essential components.

*   **Memory Directory**: Contains the SQLite database and character reference sheets.
*   **Checkpoints File**: Tracks the completion status of every file and stage.
*   **Clips Master Plan**: The `clips.json` file in the output directory, which serves as the blueprint for the entire video.
*   **Asset Folders**: The `images` and `audio` directories containing all generated media.
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = secret_value_0
print("✓ GROQ_API_KEY added to environment!")
!git clone https://github.com/bad-boy-01/NFV_v4.5-master.git
%cd NFV_v4.5-master
!python start_pipeline.py --input projects/novel/input/chapter1.txt"# factory" 
