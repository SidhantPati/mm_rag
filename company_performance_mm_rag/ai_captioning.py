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

# Instantiating Gemini Model
llm = ChatGoogleGenerativeAI(
    model=gemini_model,
    max_output_tokens=500)

# Create a chain that automatically parses the output into a pure string
lcel = (llm | StrOutputParser())

def image_captioning(image_base64):
    # 1. Construct the multimodal message content list
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

    # 2. Wrap it in a LangChain HumanMessage
    message = HumanMessage(content=message_content)

    # 3. Invoke the model
    response = lcel.invoke([message])
    return response

def table_captioning(table_as_html):
    # 1. Construct the multimodal message content list
    message_content = [
        {
            "type": "text",
            "text": (
                "You are an expert data analyst and retrieval-augmented generation (RAG) specialist. "
                "Analyze the provided table and write a thorough, information-dense summary "
                "designed to be indexed in a semantic vector database.\n\n"
                
                "Strictly follow these guidelines in your summary:\n"
                "1. **Overview & Scope:** Clearly state what the table represents, including the entities "
                "being compared, time periods (e.g., fiscal years, quarters), and units of measurement (e.g., USD in millions, percentages).\n"
                "2. **Explicit Relationship Mapping:** Always couple row labels with their corresponding column headers and values "
                "(e.g., write 'For Cloud Services, Q3 revenue was $12.4M' rather than just listing raw numbers).\n"
                "3. **Exact Metric Retention:** Preserve key numerical values, financial figures, growth rates, "
                "and performance metrics without generalizing or rounding away critical data.\n"
                "4. **Notable Trends & Outliers:** Highlight peak performers, lowest values, significant variances, "
                "or critical conclusions evident in the data.\n\n"
                
                "Output your response strictly as continuous, natural language paragraphs. Do not use markdown tables or bulleted lists, "
                "as smooth narrative prose yields superior retrieval performance for vector embeddings."
)
        },
        {
            "type": "text",
            "text": f"Here is the table content in HTML format to analyze:\n\n{table_as_html}"
        }
    ]

    # 2. Wrap it in a LangChain HumanMessage
    message = HumanMessage(content=message_content)

    # 3. Invoke the model
    response = lcel.invoke([message])
    return response