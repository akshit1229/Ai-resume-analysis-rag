import sys
import streamlit as st
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import run_ingestion
from rag_engine import ResumeRAG

st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("📄 Expert AI Resume Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    if os.path.exists("./db"):
        st.session_state.rag_engine = ResumeRAG()
    else:
        st.session_state.rag_engine = None

with st.sidebar:
    st.header("Data Management")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        upload_dir = "./resumes"
        if not os.path.exists(upload_dir): os.makedirs(upload_dir)
        for file in uploaded_files:
            with open(os.path.join(upload_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files!")
    
    if st.button("-- Process/Ingest Resumes"):
        with st.spinner("Processing... This may take a while for large batches."):
            run_ingestion()
        st.success("Ingestion Complete!")
        st.session_state.rag_engine = ResumeRAG()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reasoning" in message:
            with st.expander("🔍 Analysis Details"):
                st.write(f"**Intent:** {message['reasoning'].get('intent', {}).get('type', 'Unknown')}")
                st.write("**Sources/Context:**")
                for src in message["reasoning"]["sources"]:
                    st.text(src)

if prompt := st.chat_input("Ask: 'List Python devs', 'Compare Harish and Amit', 'Who has 10+ years exp?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.rag_engine is None:
            st.error("------- No database found. Please upload resumes and click Process!")
        else:
            with st.spinner("Analyzing candidate database..."):
                response_text, intent, docs = st.session_state.rag_engine.answer_query(prompt)
                
                # Helper to extract source names safely
                source_names = []
                for doc in docs:
                    if isinstance(doc, str): # If we passed raw strings in List mode
                        source_names.append(doc.split('\n')[0]) # Just grab the first line (Name)
                    else: # If it's a Document object
                        source_names.append(doc.metadata.get('source', 'Unknown'))
                
                # Remove duplicates
                source_names = list(set(source_names))
                
                st.markdown(response_text)
                
                reasoning_data = {
                    "intent": intent,
                    "sources": source_names
                }
                
                with st.expander("🔍 Analysis Details"):
                    st.write(f"**Intent Detected:** `{intent.get('type')}`")
                    st.write("**Sources Identified:**")
                    for src in source_names:
                        st.text(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "reasoning": reasoning_data
    })