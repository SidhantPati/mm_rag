import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_model = "gemini-3.5-flash-lite"
text_embedding_model = "BAAI/bge-large-en-v1.5"
vector_db_name = "chroma_vector_db"

# Load environment variables
load_dotenv()

# Verification to authenticate API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY is not set in the environment or .env file.")
else:
    print("API key loaded successfully!")

# Connect to Chroma Vector db and initialize retriever
embeddings = HuggingFaceEmbeddings(model_name=text_embedding_model)
vectorstore = Chroma(
        persist_directory=vector_db_name, 
        embedding_function=embeddings
        )
retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=3)

# Initializing Gemini API Call
llm = ChatGoogleGenerativeAI(
    # temperature = 0,
    model = gemini_model)

system_prompt = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Use only the given context to answer any question.
If you don't know the answer, say you dont know but dont give vague response.
Context:
{context}
"""

def retrieved_context_from_vector_db(user_prompt):
    retrieved_context = retriever.invoke(user_prompt)
    text_context = "\n\n".join(doc_chunk.page_content for doc_chunk in retrieved_context)
    return text_context

def llm_response(user_prompt):
    text_context = retrieved_context_from_vector_db(user_prompt)
    enriched_system_prompt = system_prompt.format(context = text_context)
    response = llm.invoke([
        SystemMessage(content=enriched_system_prompt),
        HumanMessage(content=user_prompt)
        ])
    response_text = "".join(item["text"] for item in response.content if item.get("type") == "text")
    return response_text

print("\n")
print(llm_response("Who is Averi Lancaster?"))
print("\n")
