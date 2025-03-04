import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Set background color
st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
        }
        .user-message {
            background-color: #f7f7f7;
            padding: 10px;
            border-radius: 10px;
        }
        .assistant-message {
            background-color: #e5e5e5;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Create a Streamlit title
st.markdown("<h2 style='text-align: center;'>Read your PDF : Interactive Q&A using Groq API</h2>", unsafe_allow_html=True)

# Initialize the LLaMA model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create a vector database from a PDF file
def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    """
    Creates a vector database from the uploaded PDF file.

    Args:
        pdf_file (file): The uploaded PDF file.
    """
    if "vector_store" not in st.session_state:
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name

        # Initialize embeddings and loader
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        st.session_state.loader = PyPDFLoader(pdf_file_path)

        # Load text documents from PDF
        st.session_state.text_document_from_pdf = st.session_state.loader.load()

        # Split text documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)

        # Create a vector store from the document chunks
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)

# Upload PDF file
pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])

# Create vector database from uploaded PDF file
if pdf_input_from_user is not None:
    if st.button("Create the Vector DB from the uploaded PDF file"):
        if pdf_input_from_user is not None:
            create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            st.success("Vector Store DB for this PDF file Is Ready")
        else:
            st.write("Please upload a PDF file first")

# Initialize chat log
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Ask user for prompt
if "vector_store" in st.session_state:
    user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

    # Submit prompt and get response
    if st.button('Submit Prompt'):
        if user_prompt:
            if "vector_store" in st.session_state:
                # Create a document chain
                document_chain = create_stuff_documents_chain(llm, prompt)

                # Create a retriever from the vector store
                retriever = st.session_state.vector_store.as_retriever()

                # Create a retrieval chain
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Invoke the retrieval chain with the user's prompt
                response = retrieval_chain.invoke({'input': user_prompt})

                # Update chat log
                st.session_state.chat_log.append({"user": user_prompt, "assistant": response['answer']})

                # Clear input field
                st.session_state.user_prompt = ""

                # Display chat log
                for chat in st.session_state.chat_log:
                    st.write(f"<div style='background-color: #f7f7f7; padding: 10px; border-radius: 10px;'><b>User:</b> {chat['user']}</div>", unsafe_allow_html=True)
                    st.write(f"<div style='background-color: #e5e5e5; padding: 10px; border-radius: 10px;'><b>Assistant:</b> {chat['assistant']}</div>", unsafe_allow_html=True)
                    st.write("")
            else:
                st.write("Please embed the document first by uploading a PDF file.")
        else:
            st.error('Please write your prompt')
