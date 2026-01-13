
import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load Environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found.")


DB_PATH = "./db"
RESUME_DIR = "./resumes"
MAX_WORKERS = 2  
RESUMES_PER_BATCH = 10 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CONSECUTIVE_ERRORS = 4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

metadata_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=1)

error_lock = threading.Lock()
consecutive_rate_limit_errors = 0
stop_signal = False

def get_existing_files(vectorstore: Chroma) -> set:
    try:
        data = vectorstore.get(include=['metadatas'])
        existing = set()
        for meta in data['metadatas']:
            if meta and 'source' in meta:
                existing.add(os.path.basename(meta['source']))
        return existing
    except Exception:
        return set()

def flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    
    clean_meta = {}
    for key, value in meta.items():
        if isinstance(value, (list, dict)): #checks whether an object belongs to a given type or tuple of types, including inherited classes.
            clean_meta[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            clean_meta[key] = ""
        else:
            clean_meta[key] = value
    return clean_meta

def extract_metadata(text: str) -> Dict[str, Any]:
    global consecutive_rate_limit_errors, stop_signal
    if stop_signal: return None

    prompt = f"""
    You are a Resume Parser. Extract data into strict JSON format.
    
    Resume Text (first 4000 chars):
    {text[:4000]}
    
    Required JSON Fields:
    - "name": (string) Candidate Name.
    - "email": (string) Candidate Email.
    - "years_exp": (int) Total years experience (numeric only).
    - "skills": (list of strings) e.g. ["python", "java"].
    - "location": (string) City/Country.
    - "education": (list of strings) e.g. ["B.Tech CS from IIT", "MBA from IIM"].
    - "summary": (string) Short summary.
    
    Output ONLY JSON.
    """
    
    retries = 3
    base_delay = 5
    
    for attempt in range(retries):
        if stop_signal: return None
        try:
            time.sleep(2)
            response = metadata_llm.invoke(prompt)
            content = response.content.strip()
            
            with error_lock: consecutive_rate_limit_errors = 0
            
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # Basic Normalization
            if 'skills' in data and isinstance(data['skills'], list):
                data['skills'] = ", ".join([str(s).lower().strip() for s in data['skills']])
            
            return data

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                with error_lock:
                    consecutive_rate_limit_errors += 1
                    current_errors = consecutive_rate_limit_errors
                
                if current_errors >= MAX_CONSECUTIVE_ERRORS:
                    with error_lock: stop_signal = True
                    logger.critical(f"------ STOPPING: Too many rate limit errors.")
                    return None
                
                wait_time = base_delay * (2 ** attempt)
                logger.warning(f"----- Rate Limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Metadata Extraction Error: {e}")
                break

    # Fallback Metadata
    return { "name": "Unknown", "email": "Unknown", "years_exp": 0, "skills": "", "location": "Unknown", "education": "", "summary": "Failed" }

def process_single_resume(file_path: str) -> List[Document]:
    if stop_signal: return []
    try:
        filename = os.path.basename(file_path)
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        full_text = " ".join([p.page_content for p in pages])
        
        if not full_text.strip(): return []

        meta = extract_metadata(full_text)
        if meta is None: return [] # Stop signal

        meta = flatten_metadata(meta)
        meta['source'] = filename
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(full_text)
        
        docs = []
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=meta))
        return docs

    except Exception as e:
        logger.error(f"File Error {file_path}: {e}")
        return []

def run_ingestion():
    if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
    
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name="resume_store")
    
    existing_files = get_existing_files(vectorstore)
    logger.info(f"------ Found {len(existing_files)} resumes already in DB.")
    
    all_files = [f for f in os.listdir(RESUME_DIR) if f.lower().endswith('.pdf')]
    files_to_process = [os.path.join(RESUME_DIR, f) for f in all_files if f not in existing_files]
    
    if not files_to_process:
        logger.info("----- All resumes are already processed. No API calls needed. -------")
        return 

    logger.info(f"------- Found {len(files_to_process)} NEW resumes to process..........") 
    
    docs_buffer = []
    resumes_processed_in_batch = 0
    total_successful = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_single_resume, f): f for f in files_to_process}
        
        for future in as_completed(future_map):
            if stop_signal: break

            docs = future.result()
            if docs:
                docs_buffer.extend(docs)
                resumes_processed_in_batch += 1 
                total_successful += 1
                
                if resumes_processed_in_batch >= RESUMES_PER_BATCH:
                    logger.info(f"💾 Saving batch of {resumes_processed_in_batch}...")
                    vectorstore.add_documents(docs_buffer)
                    docs_buffer = []
                    resumes_processed_in_batch = 0

    if docs_buffer:
        logger.info(f"-------- Saving final batch of {len(docs_buffer)} chunks.......")
        vectorstore.add_documents(docs_buffer)
    
    if stop_signal:
        logger.critical(f"---- Process stopped due to Rate Limits. Saved {total_successful} resumes before stopping.")
    else:
        logger.info("---- Ingestion Complete!") 

if __name__ == "__main__":
    os.makedirs(RESUME_DIR, exist_ok=True)
    run_ingestion()
