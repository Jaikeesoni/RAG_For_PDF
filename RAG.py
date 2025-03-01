import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore')

# Add a model selector for flexibility
st.set_page_config(page_title="Document Q&A", page_icon="📚")

# Set up the page
st.title("📚 Document Q&A System")
st.write("Upload PDF documents and ask questions about their content!")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model provider selection
    model_provider = st.selectbox(
        "Select Model Provider:",
        ["OpenAI"]
    )
    
    # API key input based on provider
    if model_provider == "OpenAI":
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set!")
        else:
            st.warning("Please enter an OpenAI API key")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type="pdf")

# Create a temporary directory for the uploaded files
temp_dir = tempfile.mkdtemp()
file_paths = []

# Function to extract text from PDFs with page tracking
def extract_text_with_pages(pdf_files):
    text_data = []
    
    for pdf_file in pdf_files:
        # Save the uploaded file to the temp directory
        file_path = os.path.join(temp_dir, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getvalue())
        file_paths.append(file_path)
        
        try:
            # Read the PDF and extract text with page numbers
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_data.append({
                        "text": text,
                        "metadata": {
                            "source": pdf_file.name,
                            "page": page_num + 1
                        }
                    })
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
    
    return text_data

# Function to process the documents and create the vector store
def process_documents(text_data, model_provider):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = []
    for item in text_data:
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            documents.append({
                "text": chunk,
                "metadata": item["metadata"]
            })
    
    # Create embeddings and vector store based on selected provider
    if model_provider == "OpenAI":
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            
            embeddings = OpenAIEmbeddings()
            
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            return vectorstore
        except Exception as e:
            st.error(f"Error with OpenAI embeddings: {str(e)}")
            st.info("Consider using the Hugging Face option which doesn't require an API key.")
            return None
    else:  # Hugging Face
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            # Use a smaller embedding model that works offline
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            return vectorstore
        except Exception as e:
            st.error(f"Error with Hugging Face embeddings: {str(e)}")
            st.info("Try installing with: pip install sentence-transformers")
            return None

# Function to ask a question and get an answer
def ask_question(vectorstore, question, model_provider):
    # Search for relevant documents
    docs = vectorstore.similarity_search(question, k=4)
    
    # Create a custom prompt
    prompt_template = """
    You are a helpful assistant that provides accurate information based only on the provided documents.
    
    Context:
    {context}
    
    Question: {question}
    
    Important instructions:
    1. Answer the question only using information from the provided context.
    2. If the answer is not in the context, say "I don't have enough information to answer this question."
    3. Do not make up or infer information that is not directly in the context.
    4. Be concise and clear in your response.
    
    Answer:
    """
    
    if model_provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
            from langchain.chains.question_answering import load_qa_chain
            from langchain.prompts import PromptTemplate
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Set up the question-answering chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
            
            # Get the answer
            response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
            answer = response["output_text"]
        except Exception as e:
            st.error(f"Error with OpenAI processing: {str(e)}")
            st.info("Consider using the Hugging Face option which doesn't require an API key.")
            
            # Fallback to a basic response
            context_texts = [doc.page_content for doc in docs]
            combined_context = "\n".join(context_texts)
            answer = f"Couldn't process with OpenAI due to API error. Here are the relevant document sections:\n\n{combined_context[:1000]}..."
    else:  # Hugging Face
        try:
            from langchain_community.llms import HuggingFaceHub
            from langchain.chains.question_answering import load_qa_chain
            from langchain.prompts import PromptTemplate
            
            # For offline use, we'll just provide the relevant passages
            context_texts = [doc.page_content for doc in docs]
            combined_context = "\n".join(context_texts)
            
            answer = f"Here are the most relevant sections for your question:\n\n{combined_context}"
            
            # Note: For a complete solution, you would need to set up HuggingFaceHub or use a local model
            # This is simplified for demonstration purposes
        except Exception as e:
            st.error(f"Error with Hugging Face processing: {str(e)}")
            
            # Just return the context as fallback
            context_texts = [doc.page_content for doc in docs]
            combined_context = "\n".join(context_texts)
            answer = f"Relevant document sections:\n\n{combined_context[:1000]}..."
    
    # Extract source information
    sources = []
    for doc in docs:
        source = f"{doc.metadata['source']} (Page {doc.metadata['page']})"
        if source not in sources:
            sources.append(source)
    
    return answer, sources, docs

# Main application flow
if uploaded_files:
    with st.spinner("Processing documents..."):
        # Extract text with page tracking
        text_data = extract_text_with_pages(uploaded_files)
        
        if not text_data:
            st.error("No text could be extracted from the uploaded documents.")
        else:
            # Show document information
            st.sidebar.success(f"Processed {len(uploaded_files)} documents")
            
            # Process documents and create vector store
            vectorstore = process_documents(text_data, model_provider)
            
            if vectorstore:
                st.sidebar.success("Documents embedded and indexed!")
                
                # Show text extraction samples
                with st.expander("Sample of extracted text"):
                    for i, item in enumerate(text_data[:3]):  # Show first 3 samples
                        st.write(f"**Document:** {item['metadata']['source']} - **Page:** {item['metadata']['page']}")
                        st.write(item['text'][:500] + "..." if len(item['text']) > 500 else item['text'])
                        st.write("---")
                
                # Question answering interface
                st.subheader("Ask a question about your documents")
                question = st.text_input("Enter your question:")
                
                if question and not question.strip() == "":
                    with st.spinner("Thinking..."):
                        answer, sources, docs = ask_question(vectorstore, question, model_provider)
                    
                    st.subheader("Answer")
                    st.write(answer)
                    
                    st.subheader("Sources")
                    for source in sources:
                        st.write(f"- {source}")
                    
                    with st.expander("View relevant document excerpts"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Document:** {doc.metadata['source']} - **Page:** {doc.metadata['page']}")
                            st.write(doc.page_content)
                            st.write("---")
            else:
                st.error("Failed to create vector store. See error messages above.")
else:
    st.info("Please upload PDF documents to begin.")

# Clean up on app close
def cleanup():
    # Remove temporary files
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove temporary directory
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

# Register cleanup function
import atexit
atexit.register(cleanup)
