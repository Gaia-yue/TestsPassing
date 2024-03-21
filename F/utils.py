import streamlit as st
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import json
from F.FastPDF_loader import RapidOCRPDFLoader
from F.chinese_splitter import ChineseRecursiveTextSplitter


def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        loader = RapidOCRPDFLoader(pdf)
        doc = loader.load()
        print(type(doc))
        text += doc[0].page_content
    return text

def path_check(pathes, state):
    if state != None :
        return pathes.index(state)
    else:
        return None
    
def generate_vectore(docs,file_name):
    
    text = get_pdf_text(docs)
    # text split sets
    chunk_size = 1000 #设置块大小
    chunk_overlap = 100 #设置块重叠大小

    # use recursivetextsplit to get chunk
    splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    chunks = splitter.create_documents(chunks)
    # load embedddings
    model_dir = 'C:\\Users\\16122\\Desktop\\notes\\Chat_test\\embedding_model\\iic\\nlp_corom_sentence-embedding_chinese-base'
    embeddings = HuggingFaceEmbeddings(model_name=model_dir)
    # vector
    local_vector_path = f"./vectores/{file_name}"
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=local_vector_path  # 允许我们将directory目录保存到磁盘上
    )
    # 持久化向量数据库
    vectordb.persist()
    return local_vector_path, file_name




        
        


        
        