from importlib.resources import path
import glob
import os

from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_embedding_model = "BAAI/bge-large-en-v1.5"
vector_db_name = "chroma_vector_db"
delete_vector_store = False

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

# Converting chunks to embeddings
embeddings = HuggingFaceEmbeddings(model_name=text_embedding_model)
if delete_vector_store:
    print("\n Deleting existing vector database from disk...")
    Chroma(
        persist_directory=vector_db_name, 
        embedding_function=embeddings
        ).delete_collection()
    
    print("Building new vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=vector_db_name
    )
else:
    print("Loading existing vector database from disk...")
    vectorstore = Chroma(
        persist_directory=vector_db_name, 
        embedding_function=embeddings
    )

print(f"\n✅ Success!! Vectorstore created with {vectorstore._collection.count()} vectors")