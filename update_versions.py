import os

files_to_update = [
    "requirements.txt",
    "ReadME/README.md",
    "main.py",
    "models/audio_adapter.py",
    "models/llm_adapter.py",
    "models/image_adapter.py",
    "Kaggle_NVF_v5.ipynb",
    "config/default.yaml",
    "core/visual/clip_builder.py",
    "core/translation/pipeline.py",
    "core/orchestrator.py",
    "core/memory/extractor.py",
    "core/memory/database.py"
]

for file_path in files_to_update:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        new_content = content.replace("v4.5", "v5").replace("v4", "v5")
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {file_path}")
    else:
        print(f"File not found: {file_path}")
