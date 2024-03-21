import streamlit as st
import os
from F.utils import  path_check
# from F.custom_LLM import CustomLLM, CustomLLM1
from F.chain import get_question_chain, summarize, get_conversation_chain
from F.question_show import QuestionShow
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Tongyi

## llm定义
def llm():
    return Tongyi(model_name="qwen-14b-chat",temperature=0.1)    


# 页面设置
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)


### 侧边栏
with st.sidebar:
    # 知识库路径
    vectore = os.listdir("./vectores/")
    # 模型路径
    model_dict = {"gpt3":"","gpt4":"","chatglm3":"" }
    # 表单填写模型信息        
    with st.form("model_utils",border=False):
        # 选择知识库和模型
        vectore_option = st.container().selectbox(
            "选择知识库", vectore,
            key="vectoredb"
        )
        model_list = model_dict.keys()
        model_option = st.container().selectbox(
        'choose your language model',
        model_list, 
        index=path_check(model_list, st.session_state.model_option)
        )
        submit_model_utils_button = st.form_submit_button("载入模型", type="primary", help="点击确认信息来载入模型")
    
    
        if submit_model_utils_button:
            # 获取知识库文件名称来得到知识库
            file_name = f"./vectores/{vectore_option}"
            st.write(file_name) 
            # 加载文本embeddings模型
            model_dir = 'C:\\Users\\16122\\Desktop\\notes\\Chat_test\\embedding_model\\iic\\nlp_corom_sentence-embedding_chinese-base'
            embeddings = HuggingFaceEmbeddings(model_name=model_dir)
            vectordb = Chroma(
            persist_directory=file_name, 
            embedding_function=embeddings
            )
            # 模型加载
            llm = llm()
            st.session_state.generate_question_chain = get_question_chain(
                vectordb=vectordb,
                llm=llm)
            st.session_state.conversation_chain = get_conversation_chain(
                vectorstore=vectordb,
                llm=llm
            )
        
                
# 题目问答布局
col1, col2 = st.columns([3, 5])

# 题目区域
#  
if st.session_state.generate_question_chain!=None and st.session_state.conversation_chain!=None:
    with col1:
        with st.container(border=True,height=750):
            QuestionShow()
            
    # 对话 记笔记
    with col2:

        message_container = st.container(height=250,border=True)

        for message in st.session_state.messages:
            message_container.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.container().chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            message_container.chat_message("user").markdown(prompt)


            response = st.session_state.conversation_chain.run(prompt)
            message_container.chat_message("assistant").markdown(response)    
            st.session_state.messages.append({"role": "assistant", "content": response})
                    
        # 记笔记
        text = st.container(border=True,height=380).text_area(height=300,label="------------------------------------------------------输入你的笔记----------------------------------------------------------")
        st.download_button('Download notes', text, file_name="notes.md")

else:
    st.warning("请先选择知识库和模型!",icon="⚠️")
        

