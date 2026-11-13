import base64
from io import BytesIO
from PIL import Image

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_model = "gemini-3.5-flash-lite"

# Load environment variables
load_dotenv()

# Verification to authenticate API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY is not set in the environment or .env file.")
else:
    print("API key loaded successfully!")

def image_captioning(image_base64):
    # 1. Instantiating Gemini Model
    llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        max_output_tokens=500)
    
    # 2. Construct the multimodal message content list
    message_content = [
        {
            "type": "text",
            "text": (
               "You are an expert data analyst and retrieval-augmented generation (RAG) assistant. "
                "Analyze this image (chart, graph, diagram, or infographic) thoroughly and generate a "
                "structured, information-dense summary optimized for semantic vector search.\n\n"
                
                "Include the following elements in your summary:\n"
                "1. **Title & Context:** What is the subject or title of the graphic? What core question does it answer?\n"
                "2. **Visual Components:** State the chart type (e.g., bar chart, line graph, flow diagram), axis labels, legends, and units of measurement.\n"
                "3. **Key Data Points & Trends:** Extract exact numerical values, percentages, peaks, troughs, growth trajectories, or comparative differences shown.\n"
                "4. **Actionable Insights:** Summarize the primary takeaways, conclusions, or business implications.\n\n"
                
                "Write the response in clear, continuous natural language paragraphs so it embeds effectively "
                "into a vector database without relying on markdown tables."
            )
        },
        {
            "type": "image_url",
            # Format the base64 string as a standard data URL
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }
    ]

    # 3. Wrap it in a LangChain HumanMessage
    message = HumanMessage(content=message_content)

    # 4. Invoke the model
    response = llm.invoke([message])
    return response.content