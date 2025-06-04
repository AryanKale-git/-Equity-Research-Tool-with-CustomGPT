import streamlit as st
import database
import bcrypt
import pickle
import time
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from secret_key import sec_key

# Set up environment variable for HuggingFace API
os.environ['HUGGINGFACEHUB_API_TOKEN'] = sec_key

# Initialize the HuggingFace model and ChatGroq for chatbot
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    model_kwargs={"max_length": 128, "token": "YOUR_HUGGINGFACE_API_KEY"}
)


chat_llm = ChatGroq(
    temperature=0,
    groq_api_key=' ',
    model="llama-3.3-70b-versatile"
)

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("Equity Research Tool & Chatbot 🔍🤖")

# Login and Signup Logic
def login():
    username = st.text_input("Username", "")
    email = st.text_input("Email (must be a Gmail)", "")
    password = st.text_input("Password", "", type="password")
    login_button = st.button("Login")

    if login_button:
        if database.login_user(username, email, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.email = email
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username, email, or password.")

def signup():
    username = st.text_input("Create Username", "")
    email = st.text_input("Enter your Email (@gmail.com only)", "")
    password = st.text_input("Create Password", "", type="password")
    signup_button = st.button("Signup")

    if signup_button:
        if not email.endswith("@gmail.com"):
            st.error("Invalid email. Please use a valid '@gmail.com' email.")
        elif database.register_user(username, email, password):
            st.success(f"Account created for {username}. Please log in.")
        else:
            st.error("Username or email already exists.")

def forgot_password():
    email = st.text_input("Enter your Email to Reset Password", "")
    new_password = st.text_input("Enter New Password", type="password")
    reset_button = st.button("Reset Password")

    if reset_button:
        if database.reset_password(email, new_password):
            st.success("Password reset successfully! Please log in.")
        else:
            st.error("Email not found!")

# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    signup()
    forgot_password()
else:
    st.sidebar.success(f"Welcome {st.session_state.username}!")

    # **Main Content - Only after login**
    # Left Side: Equity Research Tool
    with st.sidebar:
        st.header("Equity Research")
        st.subheader("News Articles URLs")
        urls = [st.text_input(f"URL {i + 1}") for i in range(3)]
        process_url_click = st.button("Process URLs")

    file_path = "faiss_vectordb.pkl"
    main_placeholder = st.empty()

    if process_url_click:
        loader = WebBaseLoader(urls)
        main_placeholder.caption("Data loading started...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=0
        )
        main_placeholder.caption("Text splitting started...")
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings()
        vectorstore_index = FAISS.from_documents(docs, embeddings)
        main_placeholder.caption("Embedding vector started...")
        time.sleep(2)

        with open(file_path, 'wb') as f:
            pickle.dump(vectorstore_index, f)

    query = main_placeholder.text_input("Ask a question about the articles:")
    if query:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain.invoke({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources")
                    for source in sources.split("\n"):
                        st.write(source)

    # Right Side: Chatbot
    with st.sidebar:
        st.header("Chatbot")
    chat_placeholder = st.empty()
    user_input = chat_placeholder.text_input("Type your question for the chatbot:")

    if user_input:
        response = chat_llm.invoke(user_input)
        st.subheader("Chatbot Response")
        st.write(response.content)

