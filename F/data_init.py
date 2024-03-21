import streamlit as st
import os

def init_sesssion():
    os.environ["DASHSCOPE_API_KEY"] = "sk-6d181d984ea941639b3fa517cc22839b"
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
    
    # 屏蔽 model 测试按钮
    if "model_button" not in st.session_state:
        st.session_state.model_button = False
    
    # 模型信息
    if "messages" not in st.session_state:
        st.session_state.messages = []


        
    if 'genre' not in st.session_state:
        st.session_state.genre = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
