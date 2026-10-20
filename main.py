from importlib.resources import path
import glob
import os

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

print(f"\n✅ Success!! Loaded {len(knowledge_base)} documents")

# Text Chunking involves the following:
# 1. Splits each Document object (i.e. parent) into multiple smaller Document 
# objects (i.e. child) based on character chunk size
# 2. Each child Docuement object chunk has samme metadata as parent metadata
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Number of characters to overlap between chunks
)

chunks = text_splitter.split_documents(knowledge_base)

print(f"\n✅ Success!! Total chunks created: {len(chunks)}\n")

