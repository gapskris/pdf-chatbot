#(PDF Reader + Chunking + Embedding)

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import os

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

#def get_vector_store(chunks):
#    embeddings = OpenAIEmbeddvings()
#    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
#    return vectorstore

    
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def get_vector_store(chunks):
    # Use Ollama embeddings instead of OpenAI
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # you can swap model name
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore