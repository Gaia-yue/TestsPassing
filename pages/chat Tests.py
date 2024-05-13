import streamlit as st
import os
import json
from F.Base_knowledge import KnowledgeBasedChatLLM


@st.cache_resource
def init_model(model_type):
    print("init_model")
    knowladge_based_chat_llm = KnowledgeBasedChatLLM()
    knowladge_based_chat_llm.init_model_config(model_type)
    return knowladge_based_chat_llm
# 页面设置
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.page_link("main.py", label="知识库加载")
st.sidebar.page_link("pages/chat Tests.py", label="考试助手")
st.sidebar.page_link("pages/history.py", label="历史题目记录")

model_type = st.sidebar.selectbox("模型类型",("Qwen2","Llama","Qwen"))
vectore_list = os.listdir("./vectores/")
vectore_option = st.sidebar.selectbox(
            "选择知识库", vectore_list,
            key="vectoredb",
            index = None
        )
# 初始化模型配置
konwledge_base_customLLM = init_model(model_type)
init_vectore = st.sidebar.button("加载知识库")
if init_vectore:
    if st.session_state.vectoredb == None:
        st.warning("请选择知识库")
    else:  
        file_name = f"./vectores/{st.session_state.vectoredb}"
        konwledge_base_customLLM.init_knowledge_vector_store(file_name)

# 对话参数
max_token = st.sidebar.slider(
    'max_token', 256, 8096, 512, step=64
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)

temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)
if st.sidebar.button("清除历史题目"):
    st.session_state.question_history = []

if st.sidebar.button("清除历史提问"):
    st.session_state.memory.clear()
# 布局
col1, col2 = st.columns([3, 5])

if konwledge_base_customLLM.ok_chat():
    with col1:
        # 题目区域  
        with st.container(border=True,height=750):
            with st.form(key="form"):
                theme = st.text_input("请输入主题", key="theme")
            
                if st.session_state.question_history == []:
                    submit_button = st.form_submit_button("开始出题",help="点击开始出题",type="primary")
                else:
                    submit_button = st.form_submit_button("继续出题",help="点击继续出题",type="secondary")

                if  submit_button:

                    if st.session_state.theme == None:
                        st.info("请选择主题!")
                    else:
                        # timu = konwledge_base_customLLM.get_question_chain_anwser(theme=theme,history="".join(st.session_state.question_history))
                        timu = konwledge_base_customLLM.get_question_chain_anwser(theme=theme,history="")
                        # st.write(timu["result"].strip('```json\n'))
                        if model_type == "Qwen2":
                            st.session_state.question = json.loads(timu["result"].strip('```json\n'))
                        else :
                            st.session_state.question = json.loads(timu["result"])
                        st.session_state.question_history_show.append(st.session_state.question)
                        st.session_state.question_history.append(str(st.session_state.question["Q_name"]))

            if st.session_state.question != None:
                Q_name = st.session_state.question["Q_name"]
                detials = st.session_state.question["detail"]
                right_Answear = st.session_state.question["right_answer"]
                t_options = st.session_state.question["options"]
                st.markdown(Q_name)
                # st.write(right_Answear)
                genre = st.radio(
                        # label="  What's your favorite movie genre?",
                        label=" please choose one",
                        options=("A","B","C","D"),
                        index=None,
                        format_func=lambda x: t_options.get(x),
                        key='genre'
                        )
                if genre == None:
                        pass
                elif genre == right_Answear:
                    with st.container():
                        st.write('正确！')
                        st.markdown("#### 问题详解")
                        with st.expander("展开来查看"):
                            st.markdown(detials)
                else:
                    with st.container():
                        st.write('错误！')
                        st.markdown("#### 问题详解")
                        with st.expander("展开来查看"):
                            st.markdown(detials)
            
    with col2:
        # 对话
        message_container = st.container(height=250,border=True)

        for message in st.session_state.messages:
            message_container.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.container().chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            message_container.chat_message("user").markdown(prompt)

            with st.spinner("typing..."):
                # response = konwledge_base_customLLM.get_conversation_chain_answser(question=prompt,memory=None,max_token=max_token,temperature=temperature,top_p=top_p)
                response = konwledge_base_customLLM.get_conversation_chain_answser(question=prompt,memory=st.session_state.memory,max_token=max_token,temperature=temperature,top_p=top_p)
                message_container.chat_message("assistant").markdown(response)    
            st.session_state.messages.append({"role": "assistant", "content": response})
                    
        # 记笔记
        text = st.container(border=True,height=380).text_area(height=300,label="------------------------------------------------------输入你的笔记----------------------------------------------------------")
        st.download_button('Download notes', text, file_name="notes.md")

else:
    st.warning("请先选择知识库和模型!",icon="⚠️")
        

