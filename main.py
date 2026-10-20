from importlib.resources import path
import glob
import os

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Store LangChain Document Objects for RAG.
# Each Document object comprises of page_content and metadata section.
loader = DirectoryLoader(
    path = "knowledge-base", 
    glob="**/*.md",
    recursive=True,
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'})
knowledge_base = loader.load()

# Enrich Document objects with additional metadata
for doc in knowledge_base:
    doc.metadata["source_type"] = Path(doc.metadata["source"]).parent.name
    doc.metadata["file_name"] = Path(doc.metadata["source"]).stem

print(f"\nSuccess!! Loaded {len(knowledge_base)} documents\n")