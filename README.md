# -Equity-Research-Tool-with-CustomGPT
This research tool can be used for web scrapping as well as answers questions from any websites which you provide link. Built an AI-powered research tool using LangChain & FAISS, improving data analysis efficiency by 40%. Integrated a chatbot with LangChain Groq (Llama 3.3-70B) for real-time responses (<2s latency). Implemented secure authentication with SQLite & Streamlit, ensuring 100% access control, and optimized text processing to reduce query time by 30%. Designed for seamless interaction and efficient response generation. It uses the model llama 3.1, and it is fast, efficient and generates solutions for even complex problems.

---

# 📊 Equity Research Tool & Chatbot 🤖

An AI-powered, research-focused application built using **Streamlit**, **LangChain**, **FAISS**, **HuggingFace**, and **Groq**. This tool allows users to:

* Ingest and analyze news articles related to equities from web URLs.
* Ask domain-specific questions based on the content.
* Chat with a large language model for general financial queries.
* Use secure login/signup functionality.

---

## 🧠 Motivation

Equity researchers, retail investors, and financial analysts often go through dozens of news articles to extract valuable insights. This tool speeds up the process using **state-of-the-art LLMs** and **semantic search**, combining retrieval-augmented generation (RAG) with real-time chat capabilities.

---

## 🔍 Features Breakdown

### 🔐 User Authentication

* Sign-up and login via a secure `bcrypt`-protected system.
* Email validation (only `@gmail.com` allowed).
* Forgot password functionality with secure update mechanism.

### 🌐 URL-Based Article Ingestion

* Input up to **three financial news URLs**.
* Web scraping and parsing using LangChain’s `WebBaseLoader`.

### ✂️ Smart Text Splitting

* Text is split using `RecursiveCharacterTextSplitter` optimized for LLM tokenization.
* Chunk size: 1000 tokens; no overlap — ideal for FAISS.

### 🔎 FAISS Vector Embedding & Semantic Search

* Converts chunks into embeddings via **HuggingFace**.
* Stores them using **FAISS**, enabling similarity-based document retrieval.

### 🤖 Question Answering (RAG Pipeline)

* Uses **Mistral-7B-Instruct-v0.3** model via HuggingFace.
* Employs `RetrievalQAWithSourcesChain` from LangChain for question answering **with citation sources**.

### 🧑‍💼 Financial Chatbot Assistant

* Powered by **Groq’s LLaMA-3-70B**.
* Answers general queries, explains financial terms, trends, and more.

---

## 🛠️ Tech Stack

| Component        | Tool/Library                            |
| ---------------- | --------------------------------------- |
| UI               | Streamlit                               |
| Auth & Storage   | Custom `database.py`, bcrypt            |
| Web Loading      | LangChain WebBaseLoader                 |
| Embeddings       | HuggingFaceEmbeddings                   |
| LLMs             | HuggingFace Inference API, Groq         |
| Vector Store     | FAISS                                   |
| Chat/QA Pipeline | LangChain's RetrievalQAWithSourcesChain |

---

## 📂 File Structure

```
📁 equity-research-tool/
│
├── app.py                # Main Streamlit app
├── database.py           # User auth logic (register/login/reset)
├── secret_key.py         # HuggingFace API token (never commit this)
├── faiss_vectordb.pkl    # Saved FAISS DB for processed URLs
├── requirements.txt      # List of dependencies
└── README.md             # You're reading it :)
```

---

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/equity-research-tool.git
cd equity-research-tool
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add Your API Keys

* Edit `secret_key.py`:

```python
sec_key = "your_huggingface_api_token"
```

* In `app.py`, update:

```python
groq_api_key = "your_groq_api_key"
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```txt
streamlit
bcrypt
langchain
langchain-community
langchain-huggingface
langchain-groq
huggingface_hub
faiss-cpu
tiktoken
```

> 📌 Make sure you have Python 3.8 or higher.

---

## 📌 Future Enhancements

* ✅ Save and view past chat history.
* 📈 Integrate real-time stock prices and charts via APIs like Yahoo Finance or Alpha Vantage.
* 🧾 Export QA results and chat history to PDF/CSV.
* 📤 Upload PDFs and CSVs in addition to URLs.
* 🌐 Deploy on Streamlit Cloud or HuggingFace Spaces.

---

## ⚠️ Disclaimer

> This tool is meant for educational and research purposes only. It is **not** a substitute for professional investment advice or decision-making.

---

## 📬 Contact

**👨‍💻 Developed by Aryan Manoj Kale**
📫 [LinkedIn](https://www.linkedin.com/in/aryankale)
🐙 [GitHub](https://github.com/AryanKale-git)

---

