import shutil
from pathlib import Path

from mm_vector_ingestion import vector_db_path
from mm_vector_ingestion import text_embedding_model
from langchain_core import embeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent # /.../mm_rag/company_performance_mm_rag/
file_path = BASE_DIR / "dataset" / "NovaCore_Multimodal_Company_Report_2026.pdf"
vector_db_path = BASE_DIR / "cp_mm_chroma_vector_db"
text_embedding_model = "BAAI/bge-large-en-v1.5"
gemini_model = "gemini-3.5-flash-lite"

# Load environment_variables
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = text_embedding_model)

vectorstore = Chroma(
    persist_directory= str(vector_db_path),
    embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=3)

llm = ChatGoogleGenerativeAI(model = gemini_model)

system_prompt = """
You are a helpful assistant answering questions about the 
NovaCore Systems FY2026 company report.
Use ONLY the retrieved context below.
If the answer is not available in the context, say:
"I could not find that information in the report."

CONTEXT:
{context}
"""
message_content = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{user_prompt}")
])

lcel = (message_content | llm | StrOutputParser())

def format_retrieved_context(docs) -> str:
    """
    Formats each retrieved document into your custom layout:
    page_content
    page_number | Chunk_number | modality
    table_as_html (if present)
    """
    formatted_chunks = []
    
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        
        # Extract metadata fields with fallbacks
        page_number = metadata.get("page_number", "N/A")
        chunk_number = metadata.get("chunk_number", "N/A")
        modality = metadata.get("modality", "N/A")
        table_html = metadata.get("text_as_html", None)
        
        # Build the structured block for this specific chunk
        lines = [
            doc.page_content,
            f"\nPage:{page_number} | Chunk: {chunk_number} | Modality: {modality}\n"
        ]
        
        if table_html:
            lines.append(table_html)
            
        formatted_chunks.append("\n".join(lines))
        
    # Join all formatted chunks together separated by clear chunk boundaries
    return "\n\n---\n\n".join(formatted_chunks)

def response_generation(user_prompt,chat_history):
    retrieved_doc_chunk = retriever.invoke(user_prompt) # return Langchain Documents
    # retrieved_context = ['\n\n'.join(doc.page_content) for doc in retrieved_doc_chunk]
    retrieved_context = format_retrieved_context(retrieved_doc_chunk)
    response = lcel.invoke({
        "context":retrieved_context,
        "chat_history":chat_history,
        "user_prompt":user_prompt
        })
    return response, retrieved_context