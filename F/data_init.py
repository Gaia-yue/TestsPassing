import streamlit as st
import os
from langchain.memory import ConversationBufferMemory

def init_sesssion():
    # 组件信息初始化
    if "vectoredb" not in st.session_state:
        st.session_state.vectoredb = None
    if "model_option" not in st.session_state:
        st.session_state.model_option = None
    
    # chains
    if "generate_question_chain" not in st.session_state:
        st.session_state.generate_question_chain = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    # chain input 模型输入
    if "question_history" not in st.session_state:
        st.session_state.question_history = []
    if "theme" not in  st.session_state:
        st.session_state.theme = None
    if 'question' not in st.session_state:
        st.session_state['question'] = None
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 屏蔽 model 测试按钮
    if "model_button" not in st.session_state:
        st.session_state.model_button = False
    
    # 模型信息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    ## 历史题目信息
    if "question_history_show" not in st.session_state:
        st.session_state.question_history_show = []
        
        
    if 'genre' not in st.session_state:
        st.session_state.genre = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    