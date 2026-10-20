from importlib.resources import path
import glob
import os

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

KNOWLEDGE_BASE_FOLDER_PATH = glob.glob("knowledge-base/*") # list fo relative folder paths
KNOWLEDGE_BASE = [] # store LangChain Document Objects for RAG

for folder_path in KNOWLEDGE_BASE_FOLDER_PATH:
    folder_name = os.path.basename(folder_path)
    loader = DirectoryLoader(
        path = folder_path, 
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata['source_type'] = folder_name
        KNOWLEDGE_BASE.append(doc)

print(f"\nSuccess!! Loaded {len(KNOWLEDGE_BASE)} documents\n")