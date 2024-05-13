from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.chains.summarize import LLMChain ,StuffDocumentsChain, ReduceDocumentsChain,MapReduceDocumentsChain
from langchain.chains import SequentialChain
from langchain.prompts import  PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from F.RetrievalQusetion import RetrievalQuestion


def get_question_chain(vectordb,llm):
    '''
    这样使用chain(theme, question_history)
    其中传入时:
    theme:     "question": str
    question_history:    "timu_history":[str]
    形式: ["question": theme, "timu_history":question_history]
    '''
    
    prompt_template =  """请结合上下文和过去的题目来进行一个单项选择题出题, 主要为逻辑题和概念题. 
    对于题目要求有:
    1. 不要和过去的题目(如果有)概念或者逻辑重复, 出题概念和方式请随机选择.
    2. 一切概念都来自上下文, 不要试图编造概念.
    3. 输出包括题目主干, 四个选项, 正确答案, 题目的详解.
    4. 请严格按照规则来输出, 输出格式为可解析json格式数据, 示例:
    {{ "Q_name" : "题目主干","options" : {{"A":"选项内容1","B":"选项内容2","C":"选项内容3","D":"选项内容4" }},"right_answer" : "ABCD中的一个，最符合Q_name的答案","detail": "题目的详解" }}
    5. 只用输出可解析json格式数据, 不要说其他的话. 
    上下文: {context}
    过去的题目:{history}
    主题:{theme}
    给出题目的json格式:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "theme", "history"]
    )
    Question_generation = RetrievalQuestion.from_llm(llm=llm,  retriever=vectordb.as_retriever(search_type="mmr",search_kwargs={'k': 10}), prompt=PROMPT)
    return Question_generation


def summarize(llm):
    map_template = map_template = """以下是包含书本知识的文档
    {docs}
    请根据这份文档,提取出包含书本章节信息的关键字和内容,字数尽可能简短/
    内容："""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt,output_key="information")

    # Reduce
    reduce_template = """
    以下是包含章节信息的内容\n
    请根据给出的书本章节信息关键字,将关键字信息以可解析json格式数据输出:
    包含下列键:书本名字,章节信息:[章节信息1, 章节信息2...]\n
    \n书本章节信息:{information}
    """
    reduce_prompt = PromptTemplate(template=reduce_template,input_variables=["information"])
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="information"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=2000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
        output_key="theme"
    )
    return map_reduce_chain

"""
e.g:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# 创建一个 PyPDFLoader Class 实例，输入为待加载的pdf文档路径
loader = PyPDFLoader("LLM-v1.0.0.pdf")
# 调用 PyPDFLoader Class 的函数 load对pdf文件进行加载
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 8000,  # 每个文本块的大小。这意味着每次切分文本时，会尽量使每个块包含 1500 个字符。
    chunk_overlap = 500  # 每个文本块之间的重叠部分。
)
splits = text_splitter.split_documents(docs)
"""

    
def get_conversation_chain(vectorstore, llm):
    """
    输入包含:
    1. 当前题目
    2. 提出的问题
    输入格式: ["question": question, "timu":timu]
    """


    # 结合文档
    document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
    document_variable_name = "context"
    reduce_template = """
    请结合上下文来回答问题, 下面是回答问题的一些要求:
    1. 如果回答包含标题,粗体,表格, 请尽量使用markdown格式
    上下文:{context}/
    问题:{question}
    """
    reduce_prompt = PromptTemplate(template=reduce_template,input_variables=["context","question"])
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name
    )

    # 压缩问题
    combine_template = """结合问答历史和提出的问题生成一个
    独立的新问题, 新问题侧重于提出的问题的概念.
    /提问历史:{chat_history}/
    提出的问题:{question}
    """
    condense_prompt = PromptTemplate(template=combine_template,input_variables=["chat_history","question"])
    question_generator_chain = LLMChain(llm=llm, prompt=condense_prompt)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain(
    combine_docs_chain=combine_documents_chain,
    question_generator=question_generator_chain,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    )

    return conversation_chain