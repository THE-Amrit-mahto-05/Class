import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

query_cache_store = None 

app = FastAPI(title="RAG Application")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = None  # will hold the FAISS index after upload


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader=PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader=TextLoader(file_path,encoding='utf-8')
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()
    # TODO 


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
    # TODO 


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=store.as_retriever(search_kwargs={"k": 3} ),
        chain_type="stuff",
        return_source_documents=True)
    
    
    # TODO 

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()
    # TODO 


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    # 1. Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # 2. Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    # 3. Sanitize filename
    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name

    # 4. Save file
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")

    # 5. Load + Build vector store
    try:
        documents = load_document(str(file_path))
        vector_store = build_vector_store(documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # 6. Return response
    return {
        "message": f"'{safe_name}' uploaded and indexed successfully.",
        "pages": len(documents)
    }
        
    # TODO


@app.post("/query")
async def query_document(req: QueryRequest):
    global query_cache_store, vector_store
    
    # 1. Check if document exists first
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first."
        )

    # 2. Initialize cache if needed
    if query_cache_store is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize with a dummy entry to avoid empty store errors
        query_cache_store = FAISS.from_texts(["placeholder"], embeddings, metadatas=[{"answer": "placeholder"}])

    # 3. Check for semantic similarity
    # FAISS distance: Lower is better (0.0 is perfect match)
    results = query_cache_store.similarity_search_with_score(req.question, k=1)
    
    if results and results[0][1] < 0.2:
        return {"answer": results[0][0].metadata["answer"], "cached": True}

    # 4. If no cache match, run QA chain
    try:
        chain = get_qa_chain(vector_store)
        result = chain.invoke({"query": req.question})

        # 5. Store result in cache
        query_cache_store.add_texts([req.question], metadatas=[{"answer": result["result"]}])

        # 6. Return response with sources
        sources = [{"content": doc.page_content[:300], "metadata": doc.metadata} 
                   for doc in result.get("source_documents", [])]

        return {
            "answer": result["result"],
            "sources": sources,
            "cached": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")