import streamlit as st
import os
import logging
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# Custom CSS for creative and attractive UI
st.markdown("""
    <style>
    body {
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2ecc71; /* Green color for heading */
        text-align: center;
        font-family: 'Playfair Display', serif; /* Stylish and attractive font */
        font-size: 36px; /* Font size */
        margin-bottom: 20px;
        font-weight: 700; /* Bold font */
    }
    .stTextInput>div>div>input {
        border: 2px solid #2ecc71; /* Green border for input */
        border-radius: 8px;
        padding: 15px;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        background-color: #f9f9f9; /* Light gray background for input */
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #27ae60; /* Darker green border on focus */
        background-color: #fff; /* White background on focus */
    }
    .stButton>button {
        background-color: #2ecc71; /* Green button background */
        color: white;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button:hover {
        background-color: #27ae60; /* Darker green for hover effect */
    }
    .stError {
        color: #e03e3e;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        padding: 10px;
    }
    .stMarkdown>p {
        color: #333;
    }
    .output-box {
        border: 2px solid #2ecc71; /* Green border for the output box */
        border-radius: 8px;
        padding: 15px;
        background-color: #f0f0f0; /* Light gray background for output */
        font-family: 'Courier New', monospace;
        color: #333; /* Text color inside the output box */
        font-weight: bold; /* Bold text */
    }
    .input-box {
        border: 2px solid #2ecc71; /* Green border for input box */
        border-radius: 8px;
        padding: 15px;
        background-color: #f9f9f9; /* Light gray background for input box */
        margin-bottom: 20px;
    }
    .input-box label {
        color: #2ecc71; /* Green color for labels */
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("CUSTOMIZE CHATBOT")

# Create a container for input fields
with st.container():
    st.markdown('<div class="input-box"><label for="url_input">Enter URL</label></div>', unsafe_allow_html=True)
    url_input = st.text_input("", key="url_input")
    
    st.markdown('<div class="input-box"><label for="prompt">Ask Your Question</label></div>', unsafe_allow_html=True)
    prompt = st.text_input("", key="prompt")

if url_input:
    try:
        st.write("WAIT, CHATBOT IS GETTING READY ")
        
        loader = WebBaseLoader(url_input)
        docs = loader.load()
        
        if not docs:
            st.error("No documents were found at the provided URL.")
        else:
            logger.info(f"Loaded {len(docs)} documents from {url_input}.")
            
            embeddings = OllamaEmbeddings()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:50])
            
            vectors = FAISS.from_documents(final_documents, embeddings)

            llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
            
            prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}
            """
            )
            
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            if prompt:
                start = time.process_time()
                response = retrieval_chain.invoke({"input": prompt})
                response_text = response.get('answer', "No answer found.")
                st.write("Response time:", time.process_time() - start)
                st.markdown(f'<div class="output-box">{response_text}</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")
