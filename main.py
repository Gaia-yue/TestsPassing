import streamlit as st
import os
from F.data_init import init_sesssion
from F.utils import generate_vectore


# state状态初始化
init_sesssion()
st.set_page_config(
    page_title="Main",
    page_icon=":icon:",
    initial_sidebar_state="expanded",
)

# 上传书本并永久本地化知识向量库
with st.container(border=True):
    st.subheader("上传知识库")
    uploaded_file = st.file_uploader("choose books", accept_multiple_files=True)
    file_name = st.text_input("请输入你的知识库名字")
    load_vec = st.button("提交")
    if load_vec:
        if  uploaded_file == []:
            st.warning("请上传文件!")
        elif file_name == "":
            st.warning("请输入知识库名字!")
        else:
            st.session_state.vectoredb, _ = generate_vectore(uploaded_file,file_name)

    # 展示已经存在的知识库
    path = "./vectores/"
    vec_list = os.listdir(path)

with st.container(border=True):
    st.subheader("存在的知识库")
    for i in vec_list:
        if i == "chroma.sqlite3":
            pass
        else:
            st.container().write(i)

# 链接
# st.page_link("main.py", label="Home", icon="🏠")
# st.page_link("pages/Chat Tests.py", label='Chat Tests', icon='💯',help="点击跳转刷题小助手") 
# st.page_link("pages/Getcode.py", label='book_load')

