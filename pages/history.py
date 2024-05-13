import streamlit as st

st.sidebar.page_link("main.py", label="知识库加载")
st.sidebar.page_link("pages/chat Tests.py", label="考试助手")
st.sidebar.page_link("pages/history.py", label="历史题目记录")
# st.sidebar.page_link("pages/codegpt.py", label="代码助手")

st.title("历史题目")

text_contents = ""

def question_show(question):
    st.markdown(question['Q_name'])
    st.write("A:",question["options"]["A"])
    st.write("B:",question["options"]["B"])
    st.write("C:",question["options"]["C"])
    st.write("D:",question["options"]["D"])
    st.write("正确答案：",question['right_answer'])
    st.write('详细解释：',question['detail'])

for i, question in enumerate(st.session_state.question_history_show):
    st.write(f"第{i+1}题")
    question_show(question)
    st.write("#################")
    text_contents += f"第{i+1}题"+"\n"+question['Q_name']+"\n"+"A:"+question["options"]["A"]+"\n"+"B:"+question["options"]["B"]+"\n"+"C:"+question["options"]["C"]+"\n"+"D:"+question["options"]["D"]+"\n"+"正确答案："+question['right_answer']+"\n"+'详细解释：'+question['detail']+"\n"+"#################"+"\n"

st.sidebar.download_button('下载历史题目', text_contents)

    



