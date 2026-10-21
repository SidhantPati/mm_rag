import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
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

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_prompt}")
])

def retrieved_context_from_vector_db(user_prompt):
    """
    Retrieve relevant context (i.e. document chunks) for a user prompt.
    """
    retrieved_context = retriever.invoke(user_prompt)
    text_context = "\n\n".join(doc_chunk.page_content for doc_chunk in retrieved_context)
    return text_context

def all_session_engagement(user_prompt, chat_history):
    """
    Combine prior user prompts with current user prompt
    """
    prior_engagement = "\n".join(chat["content"] for chat in chat_history if chat["role"]=="user")
    return prior_engagement + "\n" + user_prompt

def llm_response(user_prompt, chat_history):
    """
    Generate LLM response to a given user prompt based on RAG
    """
    full_engagement = all_session_engagement(user_prompt, chat_history)
    text_context = retrieved_context_from_vector_db(full_engagement)
    enriched_system_prompt = qa_prompt.format_messages(
        context = text_context,
        user_prompt = user_prompt)
    response = llm.invoke(enriched_system_prompt)
    response_text = "".join(item["text"] for item in response.content if item.get("type") == "text")
    return response_text
