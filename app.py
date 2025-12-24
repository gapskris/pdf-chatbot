#(User Input + Retrieval + Chat)
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from helper import load_pdf, get_chunks, get_vector_store

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Q&A Bot")
st.title("Ask Questions from your PDF 📄🤖")

pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    # Save uploaded file temporarily
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf.read())

    text = load_pdf("uploaded.pdf")
    chunks = get_chunks(text)
    vectorstore = get_vector_store(chunks)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        
        OLLAMA_BASE_URL = "http://localhost:11434/v1"
        llm = OpenAI(temperature=0,base_url=OLLAMA_BASE_URL, api_key="anything")
        #response = ollama.chat.completions.create(model="llama3.2", messages=[{"role":"user", "content": "what is 2+2?"}])
        #response = ollama.chat.completions.create(model="llama3.2", messages=message)
        #print(response.choices[0].message.content)
        #return (response.choices[0].message.content)
        
        
        #llm = OpenAI(temperature=0, openai_api_key=openai_key)
        qa_chain = RetrievalQA.from_chain_type(llm=ollama, retriever=vectorstore.as_retriever())
        answer = qa_chain.run(query)
        st.write("Answer:", answer)

