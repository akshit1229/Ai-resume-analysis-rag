
import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

load_dotenv()
api_key =  os.getenv("GROQ_API_KEY")
# Verify API Key
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

DB_PATH = "./db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class ResumeRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings,
            collection_name="resume_store"
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        
    def _get_all_metadata(self) -> List[Dict]:
        """Helper to fetch the registry of all candidates for filtering."""
        try:
            data = self.vectorstore.get(include=['metadatas'])
            unique_candidates = {}
            
            for meta in data['metadatas']:
                source = meta.get('source')
                if source and source not in unique_candidates:
                    unique_candidates[source] = meta
            
            return list(unique_candidates.values())
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return []

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classifies the user's intent."""
        parser = JsonOutputParser()
        
        prompt = PromptTemplate(
            template="""
            Analyze the User Query: "{query}"
            
            Classify into one of these types and return JSON:
            
            1. "type": "list_filter"
               - Use this when user asks to "find", "list", "show me", "count" candidates based on skills, location, education, or experience.
               - Extract "keywords" (e.g., ["Python", "Pune", "B.Tech", "5 years"]).
               
            2. "type": "compare_specific"
               - Use this when user asks about SPECIFIC people by name.
               - Extract "target_names" (list of names).
               
            3. "type": "general"
               - Use for conceptual questions.
               
            JSON Output:
            """,
            input_variables=["query"],
        )
        
        try:
            chain = prompt | self.llm | parser
            return chain.invoke({"query": query})
        except Exception:
            return {"type": "general"}

    def handle_list_query(self, query: str, intent: Dict) -> tuple:
        """Strategy: Fetch ALL metadata, filter loosely, let LLM refine."""
        all_candidates = self._get_all_metadata()
        
        candidate_summaries = []
        keywords = [k.lower() for k in intent.get('keywords', [])]
        
        for cand in all_candidates:
            cand_str = (
                f"{cand.get('name', '')} "
                f"{cand.get('skills', '')} "
                f"{cand.get('location', '')} "
                f"{cand.get('education', '')} "
                f"{cand.get('years_exp', 0)}"
            ).lower()
            
            # Filter: Check if keywords match any part of the candidate's profile
            if not keywords or any(k in cand_str for k in keywords):
                candidate_summaries.append(
                    f"- Name: {cand.get('name', 'Unknown')}\n"
                    f"  Location: {cand.get('location', 'Unknown')}\n"
                    f"  Education: {cand.get('education', 'Unknown')}\n"
                    f"  Exp: {cand.get('years_exp', 0)} years\n"
                    f"  Skills: {cand.get('skills', 'N/A')}\n"
                )

        # Limit to avoid token overflow
        context_text = "\n\n".join(candidate_summaries[:30])
        
        system_prompt = f"""
        You are an HR Analyst. The user wants a list of candidates.
        
        User Request: "{query}"
        
        Candidate Registry:
        {context_text}
        
        Your Job:
        1. Filter this list strictly based on the user's request.
        2. If the user asked for "Pune", only show candidates with Location "Pune".
        3. If the user asked for "B.Tech", only show candidates with that Education.
        4. Present the result as a clean list.
        """
        
        response = self.llm.invoke(system_prompt)
        return response.content, candidate_summaries[:10]

    def handle_compare_query(self, query: str, intent: Dict) -> tuple:
        """Strategy: Find specific documents for the named people and compare them."""
        target_names = [n.lower() for n in intent.get('target_names', [])]
        all_candidates = self._get_all_metadata()
        
        matched_files = []
        for cand in all_candidates:
            c_name = cand.get('name', '').lower()
            if any(t in c_name for t in target_names):
                matched_files.append(cand.get('source'))
        
        if not matched_files:
            return "I couldn't find resumes matching those names.", []

        docs = self.vectorstore.get(where={"source": {"$in": matched_files}})
        
        full_profiles = {}
        for meta, content in zip(docs['metadatas'], docs['documents']):
            name = meta.get('name', 'Unknown')
            if name not in full_profiles: full_profiles[name] = ""
            full_profiles[name] += content + "\n"

        context_text = ""
        for name, text in full_profiles.items():
            context_text += f"=== PROFILE: {name} ===\n{text[:3000]}\n\n"

        system_prompt = f"""
        Compare the following candidates based on: "{query}"
        
        {context_text}
        
        Provide a detailed side-by-side comparison.
        """
        
        response = self.llm.invoke(system_prompt)
        source_docs = [Document(page_content="Full Profile Used", metadata={"source": f}) for f in matched_files]
        return response.content, source_docs

    def handle_general_query(self, query: str) -> tuple:
        """Strategy: Standard Semantic Search."""
        docs = self.vectorstore.similarity_search(query, k=5)
        
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}" 
            for doc in docs
        ])
        
        system_prompt = f"Answer using this context:\n{context_text}\n\nQuestion: {query}"
        response = self.llm.invoke(system_prompt)
        return response.content, docs

    def answer_query(self, query: str):
        intent = self.classify_intent(query)
        intent_type = intent.get('type', 'general')
        
        if intent_type == 'list_filter':
            response, docs = self.handle_list_query(query, intent)
        elif intent_type == 'compare_specific':
            response, docs = self.handle_compare_query(query, intent)
        else:
            response, docs = self.handle_general_query(query)
            
        return response, intent, docs
