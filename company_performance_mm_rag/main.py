from unstructured.partition.pdf import partition_pdf

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