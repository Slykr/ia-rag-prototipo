import streamlit as st
import os
from utils.loader import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Carrega as variáveis do .env

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("💬 IA com seus documentos")
uploaded_file = st.file_uploader("Faça upload de um documento", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_path = f"./temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processando o documento..."):
        docs = load_documents(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        openai_api_key = os.getenv("OPENAI_API_KEY")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        retriever=retriever)
        
        db = FAISS.from_documents(chunks, embeddings)

        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

    st.success("Documento processado com sucesso!")

    pergunta = st.text_input("Pergunte algo sobre o documento:")

    if pergunta:
        resposta = qa.run(pergunta)
        st.write("🧠 Resposta:")
        st.info(resposta)
