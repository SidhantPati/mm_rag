from transformers.agents import DuckDuckGoSearchTool
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from collections import Counter
from langchain_core.documents import Document
from ai_captioning import image_captioning, table_captioning

file_path = 'company_performance_mm_rag/dataset/NovaCore_Multimodal_Company_Report_2026.pdf'

def pdf_element_extracter(file_path):
    print(f"\n⏳ Initiating - pdf elements extraction...")
    elements = partition_pdf(
        filename=file_path, # local file path to the PDF you want to parse
        strategy="hi_res", # underlying method used to parse the document

        # detects table grids, parses row/column cells, and exports the data as clean HTML
        infer_table_structure=True,
        
        # Specifies which element types to extract images from (e.g., ["Image", "Table"]). Works best wth "hi_res"
        extract_image_block_types=["Image"],
        
        # convert images into base64 strings and embed them directly inside each element's metadata fields
        extract_image_block_to_payload=True
    )
    print(f"\n✅ Success!! Extracted {len(elements)} elements.")
    return elements

elements = pdf_element_extracter(file_path)

def create_chunks_by_title(elements):
    """Groups atomic elements and chunks them to create Composite-Elements"""
    chunks = chunk_by_title(
        elements, # The parsed atomic PDF elements comprising of titles, narrative-text, charts, tables etc
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2500, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

chunks = create_chunks_by_title(elements)

def extract_chunk_data_content(chunk):
    data_content = {
        'text':'',
        'image':[],
        'table':[],
        'data_types_counter':{},
        'langchain_doc_list':[]
    }

    data_types = []
    doc = []

    for atomic_element in chunk.metadata.orig_elements:
        # element_type can be Image, Table, Title, NarrativeText, ListItem
        element_type = str(type(atomic_element).__name__)

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
                    'image_base64':atomic_element.metadata.image_base64,
                    'page_number':atomic_element.metadata.page_number
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
                    'table_as_html':atomic_element.metadata.text_as_html,
                    'page_number':atomic_element.metadata.page_number
                }
            ))
        
        else:
            data_content['text'] = data_content['text'] + ' \n\n '+ atomic_element.text
            data_types.append('text')
            # store text as Langchain doc
            doc.append(Document(
                page_content=atomic_element.text,
                metadata={
                    'modality':'text',
                    'page_number':atomic_element.metadata.page_number
                }
            ))
    
    # Count the frequency of each datatype and store in dictionary
    data_content['data_types_counter'] = dict(Counter(data_types))
    # Store list of all Langchain docs generated from this chunk
    data_content['langchain_doc_list'] = doc
    return(data_content)

for chunk_index, chunk in enumerate(chunks):
    chunk_number = chunk_index + 1
    processed_chunk = extract_chunk_data_content(chunk)
    print(f"------ Processed chunk {chunk_number} ------ {processed_chunk['data_types_counter']}")
    print(processed_chunk['langchain_doc_list'])
    break