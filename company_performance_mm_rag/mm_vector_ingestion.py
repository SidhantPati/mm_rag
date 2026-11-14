import os
import shutil
from pathlib import Path
from collections import Counter

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ai_captioning import image_captioning, table_captioning

BASE_DIR = Path(__file__).resolve().parent # /.../mm_rag/company_performance_mm_rag/
file_path = BASE_DIR / "dataset" / "NovaCore_Multimodal_Company_Report_2026.pdf"
vector_db_path = BASE_DIR / "cp_mm_chroma_vector_db"

text_embedding_model = "BAAI/bge-large-en-v1.5"
# Control to check if use existing vector store or delete & recreate a new vector db instance
delete_existing_vector_store = True

def pdf_element_extracter(file_path):
    """Extracts atomic multimodal elements from the PDF using unstructured."""
    elements = partition_pdf(
        filename=str(file_path), # local file path to the PDF you want to parse
        strategy="hi_res", # underlying method used to parse the document

        # detects table grids, parses row/column cells, and exports the data as clean HTML
        infer_table_structure=True,
        
        # Specifies which element types to extract images from (e.g., ["Image", "Table"]). Works best wth "hi_res"
        extract_image_block_types=["Image"],
        
        # convert images into base64 strings and embed them directly inside each element's metadata fields
        extract_image_block_to_payload=True
    )
    return elements

def create_chunks_by_title(elements):
    """Groups atomic elements and chunks them to create Composite-Elements"""
    chunks = chunk_by_title(
        elements, # The parsed atomic PDF elements comprising of titles, narrative-text, charts, tables etc
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2500, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    return chunks

def chunk_processing(chunk, chunk_number):
    """Process individual chunks
    Input: chunk
    
    Processing:
    1. Extract Atomic elements from each chunk (Title, NarrativeText, ListText, Image, Table)
    2. Generate AI Summary for non-Text elements (i.e. Image & Table)
    3. Store the textual content and metadata in a Langchain doc

    Output: data_content (Dict)
    """
    data_content = {
        'text':'',
        'image':[],
        'table':[],
        'data_types_counter':{},
        'langchain_doc_list':[]
    }

    data_types = []
    doc = []
    # Safe retrieval of page number (composite chunk page_number or fallback to 1)
    chunk_page_number = getattr(chunk.metadata, "page_number", 1) or 1

    # # 1. Primary Text Document: Use the unified composite text created by chunk_by_title
    if chunk.text:
        data_types.append('text')
        doc.append(Document(
                page_content=chunk.text,
                metadata={
                    'modality':'text',
                    'chunk_number':chunk_number,
                    'page_number':chunk_page_number
                }
            ))
    
    #2. Iterate over atomic elemnts only to extract and summarize Images & Tables
    for atomic_element in chunk.metadata.orig_elements:
        # element_type can be Image, Table, Title, NarrativeText, ListItem
        element_type = type(atomic_element).__name__

        if element_type == 'Image':
            data_content['image'].append(atomic_element.metadata.image_base64)
            data_types.append('image')
            # generate image summary
            img_summary = image_captioning(atomic_element.metadata.image_base64)
            # store image summary as Langchain doc
            doc.append(Document(
                page_content=img_summary,
                metadata={
                    'modality':'image',
                    'chunk_number':chunk_number,
                    'page_number':atomic_element.metadata.page_number or chunk_page_number
                    # 'image_base64':atomic_element.metadata.image_base64, 
                    # (Not storing img_base64 owing to large size)
                }
            ))

        elif element_type == 'Table':
            data_content['table'].append(atomic_element.metadata.text_as_html)
            data_types.append('table')
            # generate table summary
            table_summary = table_captioning(atomic_element.metadata.text_as_html)
            # store table summary as Langchain doc
            doc.append(Document(
                page_content=table_summary,
                metadata={
                    'modality':'table',
                    'chunk_number':chunk_number,
                    'table_as_html':atomic_element.metadata.text_as_html or atomic_element.text,
                    'page_number':atomic_element.metadata.page_number or chunk_page_number
                }
            ))
    
    # Count the frequency of each datatype and store in dictionary
    data_content['data_types_counter'] = dict(Counter(data_types))
    # Store list of all Langchain docs generated from this chunk
    data_content['langchain_doc_list'] = doc
    return(data_content)

# =====================================================================
# Main Execution Pipeline
# =====================================================================

if __name__ == "__main__":

    print("\n⏳ Reading data file and to extract multi-modal elements...")
    elements = pdf_element_extracter(file_path)
    print(f"✅ Success!! Extracted {len(elements)} multi-modal elements from the data file\n")

    print("⏳ Initiating Chunking of elements...")
    chunks = create_chunks_by_title(elements)
    print(f"✅ Success!! Created {len(chunks)} chunks\n")

    print("⏳ Instantiating Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=text_embedding_model)
    print(f"✅ Success!! instantiated Embedding Model of {len(embeddings.embed_query('dimension check'))} dimensions\n")

    if delete_existing_vector_store and vector_db_path.exists():
        print(f"🗑️ Removing existing database at: {vector_db_path}")
        shutil.rmtree(vector_db_path)
    print("⏳ Building new vector database...")
    vectorstore = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=embeddings
    )
    print(f"✅ Success!! Vectorstore instantiated\n")

    print("Initiating chunk processing...")
    for chunk_index, chunk in enumerate(chunks):
        chunk_number = chunk_index + 1
        processed_chunk = chunk_processing(chunk, chunk_number)
        vectorstore.add_documents(processed_chunk['langchain_doc_list'])
        print(f"\t ------ Processed chunk {chunk_number} comprising of {processed_chunk['data_types_counter']}")
    print(f"🚀 Completed!! Added {len(chunks)} chunks to the vector_db\n")