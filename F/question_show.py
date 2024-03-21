import streamlit as st
import json

def ShowHistoryQ():
    pass
def Stop():
    st.write(f"太棒了!今日你已经刷了{st.session_state.len(st.session_state.question_history)}道题了!")

def QuestionShow():
    # st.write(st.session_state.question_history)
    # st.write(st.session_state.genre)
    with st.form(key="form"):
        theme = st.text_input("请输入主题", key="theme")
        # theme = st.container().selectbox(
        # '请选择章节',
        # st.session_state.themes,
        # key='theme'
        # )
        if st.session_state.question_history == []:
            submit_button = st.form_submit_button("开始出题",help="点击开始出题",type="primary")
        else:
            submit_button = st.form_submit_button("继续出题",help="点击继续出题",type="secondary")

        if  submit_button:
            
            # 第一次 没有问题 只有主题
            # 首先生成问题
            # question输入: 主题和问题历史
            # st.session_state.question = st.session_state.question_chain(theme, "".join(st.session_state.question_history))
            # 添加question
            # st.session_state.question_history.append(qustion)
            
            # """
            # 代码健壮性测试
            # # n = 0
            # # n = n+1
            # # st.write("n:", n)
            # # st.write(st.session_state.theme)
            # """

            if st.session_state.theme == None:
                st.info("请选择主题!")
            else:
                timu = st.session_state.generate_question_chain({"theme":theme,"history":"".join(st.session_state.question_history)})
                st.session_state.question = json.loads(timu["result"])
                st.session_state.question_history.append(str(st.session_state.question["Q_name"]))

    if st.session_state.question != None:
        Q_name = st.session_state.question["Q_name"]
        detials = st.session_state.question["detail"]
        right_Answear = st.session_state.question["right_answer"]
        t_options = st.session_state.question["options"]
        st.markdown(Q_name+"?")
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
                st.write('you are right')
                st.markdown("#### 问题详解")
                with st.expander("展开来查看"):
                    st.text(detials)
        else:
            with st.container():
                st.write('you are wrong')
                st.markdown("#### 问题详解")
                with st.expander("展开来查看",):
                    st.markdown(detials)


# QuestionShow()



