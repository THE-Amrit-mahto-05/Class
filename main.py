import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="RAG Application", layout="centered")

# Initialize session state for vector store and cache so they persist across re-runs
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "query_cache_store" not in st.session_state:
    st.session_state.query_cache_store = None


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )


# --------------- UI elements ---------------

st.title("RAG — Ask Your Document")

# 1. Upload Section
st.header("1. Upload a Document")
st.markdown("Supports **.pdf** and **.txt** files.")
uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded_file is not None:
    # Save the uploaded file temporarily so PyPDFLoader/TextLoader can read it
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    try:
        with st.spinner(f"Processing and indexing '{uploaded_file.name}'..."):
            documents = load_document(temp_file_path)
            st.session_state.vector_store = build_vector_store(documents)
        st.success(f"'{uploaded_file.name}' uploaded and indexed successfully. Indexed {len(documents)} document pieces.")
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

st.divider()

# 2. Query Section
st.header("2. Ask a Question")
question = st.text_input("e.g. What is the main topic of this document?")

if st.button("Ask"):
    if st.session_state.vector_store is None:
        st.error("No document uploaded yet. Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Initialize cache if needed
        if st.session_state.query_cache_store is None:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.query_cache_store = FAISS.from_texts(
                ["placeholder"], 
                embeddings, 
                metadatas=[{"answer": "placeholder"}]
            )

        with st.spinner("Thinking..."):
            # Check cache
            results = st.session_state.query_cache_store.similarity_search_with_score(question, k=1)
            
            # FAISS distance: Lower is better. Distance of < 0.2 means a very strong semantic match.
            if results and results[0][1] < 0.2:
                answer = results[0][0].metadata["answer"]
                st.markdown("### Answer")
                st.write(answer)
                st.info("ℹ️ Answer served from cache.")
            else:
                try:
                    chain = get_qa_chain(st.session_state.vector_store)
                    result = chain.invoke({"query": question})
                    
                    answer = result["result"]
                    
                    # Store result in cache
                    st.session_state.query_cache_store.add_texts(
                        [question], 
                        metadatas=[{"answer": answer}]
                    )
                    
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    # Display Sources
                    sources = result.get("source_documents", [])
                    if sources:
                        st.markdown("#### Sources:")
                        for i, doc in enumerate(sources):
                            with st.expander(f"Source [{i + 1}]"):
                                st.write(doc.page_content[:300] + "...")
                                st.write("**Metadata:**", doc.metadata)
                                
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")