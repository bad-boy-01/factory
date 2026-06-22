import subprocess
import time
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineStarter")

def run_pipeline(input_file):
    # 1. Run setup script
    logger.info("Running kaggle_setup.sh...")
    subprocess.run(["bash", "kaggle_setup.sh"], check=True)

    # 2. Ollama setup skipped as per user request

    # 5. Run main.py
    logger.info(f"Running pipeline with input: {input_file}")
    cmd = ["python", "main.py", "novel", "--input", input_file]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="projects/novel/input/chapter1.txt", help="Path to input file")
    args = parser.parse_args()
    
    run_pipeline(args.input)
