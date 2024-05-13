import streamlit as st
import os
from F.data_init import init_sesssion
from F.utils import generate_vectore

# state状态初始化
init_sesssion()
st.set_page_config(
    page_title="知识库上传页面",
    initial_sidebar_state="expanded",
)
st.title("知识库")


st.sidebar.page_link("main.py", label="知识库加载")
st.sidebar.page_link("pages/chat Tests.py", label="考试助手")
st.sidebar.page_link("pages/history.py", label="历史题目记录")
st.sidebar.page_link("pages/codegpt.py", label="代码助手")


# 上传书本并永久本地化知识向量库
with st.container(border=True):

    uploaded_file = st.file_uploader("choose books", accept_multiple_files=True)
    file_name = st.text_input("请输入你的知识库名字")
    load_vec = st.button("提交")
    if load_vec:
        if  uploaded_file == []:
            st.warning("请上传文件!")
        elif file_name == "":
            st.warning("请输入知识库名字!")
        else:
            with st.progress("加载知识库中...."):
                st.session_state.vectoredb, _ = generate_vectore(uploaded_file,file_name)


# 展示已经存在的知识库
# 定义常量来提高代码的可维护性
VECTORS_PATH = "./vectores/"
EXCLUDED_FILE = "chroma.sqlite3"

try:
    # 使用列表推导式过滤文件列表，提高代码的简洁性
    vec_list = [f for f in os.listdir(VECTORS_PATH) if f != EXCLUDED_FILE]
except OSError as e:
    # 添加异常处理来处理os.listdir可能抛出的异常
    st.error(f"无法列出目录{VECTORS_PATH}的内容: {e}")
    vec_list = []

if vec_list:
    with st.container(border=True):
        st.subheader("存在的知识库")
        # 优化：避免在循环中重复创建容器，改为一次性写入所有文件名
        for i in vec_list:
            st.write(i)
else:
    # 处理vec_list为空的情况，确保代码的鲁棒性
    with st.container(border=True):
        st.subheader("存在的知识库")
        st.write("当前目录下没有其他文件。")


            

# 链接
# st.page_link("main.py", label="Home", icon="🏠")
# st.page_link("pages/Chat Tests.py", label='Chat Tests', icon='💯',help="点击跳转刷题小助手") 
# st.page_link("pages/Getcode.py", label='book_load')

