import os
from core.project_manager import ProjectManager

pm = ProjectManager(os.getcwd(), "novel")
print(f"Base dir: {os.getcwd()}")
print(f"Project dir: {pm.project_dir}")
print(f"Input dir: {pm.dirs['input']}")
print(f"Files in input dir: {os.listdir(pm.dirs['input'])}")
print(f"Filtered input files: {pm.get_input_files()}")
