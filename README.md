# 📄 AI Resume Analyst (RAG-based)

A powerful, local RAG (Retrieval-Augmented Generation) application designed to help HR professionals and Recruiters analyze hundreds of resumes instantly using Natural Language.

Built with **LangChain**, **Streamlit**, **ChromaDB**, and **Groq (Llama-3)**.

---

## 🚀 How It Helps

Finding the right candidate from a pile of PDFs is tedious. This tool turns your resume folder into a **Smart Database** that you can talk to.

Instead of searching for keywords manually, you can ask:
* *"Find Python developers with 5+ years of experience in Bangalore."*
* *"Compare Harish and Amit side-by-side for a Senior Developer role."*
* *"Who has a Masters degree and knows Machine Learning?"*

The system extracts structured data (Name, Email, Skills, Experience, Education, Location) from every resume and allows for **Intelligent Filtering** + **Semantic Search**.

---

## ✨ Key Features

* **🧠 Intelligent Ingestion**: Automatically reads PDFs, extracts metadata (JSON), and flattens complex fields for database storage.
* **🔍 Smart Search Router**:
    * **List Mode**: Filters candidates by strict criteria (e.g., "List Java devs").
    * **Compare Mode**: Fetches full profiles of specific candidates for detailed comparison.
    * **General Mode**: Answers conceptual questions using semantic similarity.
* **🛡️ Rate Limit Protection**: Built-in exponential backoff and safety stops to handle API rate limits (Groq Free Tier friendly).
* **📂 Duplicate Detection**: Skips already processed files to save time and API usage.
* **⚡ CPU Optimized**: Runs embedding models locally on CPU to prevent hardware conflicts.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **LLM (Reasoning)**: Groq API (Llama-3-70b)
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
* **Vector DB**: ChromaDB (Persistent local storage)
* **PDF Parsing**: PyMuPDF
* **Orchestration**: LangChain

---

## ⚙️ Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Project_name.git](https://github.com/your-username/Project_name.git)
```
### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Set Up API Keys
### 5. Run in terminal
```bash
streamlit run app.py
```
### 6. UI interface run in browser add resume and start ingestion
### 7. After completing ingestion you can chat with the chatbot.
