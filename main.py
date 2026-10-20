import glob

from pathlib import Path

KNOWLEDGE_BASE_FOLDER_PATH = glob.glob("knowledge-base/*/*.md")
KNOWLEDGE_BASE = {} # store KB data in dictionary format

for file_path in KNOWLEDGE_BASE_FOLDER_PATH:
    file_name = Path(file_path).stem # read file names without extension
    with open(file_path,"r") as f:
        KNOWLEDGE_BASE[file_name] = f.read()

print(KNOWLEDGE_BASE["Rellm"])