import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Set up Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Alternative way using Streamlit secrets
# groq_api_key = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image("ProgayanAI_Transparent.png")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Initialize session state for vector store and chat history
if "vector" not in st.session_state:
    st.session_state.vector = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    # Write file to a temporary location
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())

                    # Load the PDF
                    loader = PyPDFLoader(file.name)
                    docs.extend(loader.load())
                  # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                final_documents = text_splitter.split_documents(docs)
                
                # Use a pre-trained model from Hugging Face for embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Store documents in FAISS vector store
                st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
                
                st.success("Documents processed successfully!")
                
                # If no document is uploaded
                else:
                    st.warning("Please upload at least one document.")
                
                # Main chat interface
                st.header("Chat with your Documents")
                
                # Initialize the language model (Groq LLaMA3)
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama3-8b-8192"
                )
