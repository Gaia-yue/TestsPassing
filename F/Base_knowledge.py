from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.chains.summarize import LLMChain ,StuffDocumentsChain
from langchain.prompts import  PromptTemplate
from F.RetrievalQusetion import RetrievalQuestion
from F.custom_LLM import Qwen, Llama3
class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None
    vectore: object = None
    model_type: str = None
    def init_model_config(self,model_type):
        self.model_type = model_type 
        self.embeddings = HuggingFaceEmbeddings(
            model_name="/data/tool/text2vec-base-chinese", )
        if model_type == "Qwen":
            self.llm = Qwen()
        if model_type == "Llama":
            self.llm = Llama3()
    def ko_chat(self):
        if self.llm != None and self.vectore != None:
            return True
        else:
            return False

        # 更改成我的功能 
    def init_knowledge_vector_store(self, filepath):
        self.vectore = Chroma(
            persist_directory=filepath, 
            embedding_function=self.embeddings
            )
        
    def get_question_chain_anwser(self,theme,history,
                            max_token: int = 1500,
                            temperature: float = 0.01,
                            top_p: float = 0.1):
        '''
        这样使用chain(theme, question_history)
        其中传入时:
        theme:     "question": str
        question_history:    "timu_history":[str]
        形式: ["question": theme, "timu_history":question_history]
        '''
        
        self.llm.max_token = max_token
        self.llm.top_p = top_p
        self.llm.temperature = temperature


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
        Question_generation = RetrievalQuestion.from_llm(llm=self.llm,  retriever=self.vectordb.as_retriever(search_type="mmr",search_kwargs={'k': 10}), prompt=PROMPT)
        response = Question_generation({'theme':theme, 'history':history})
        return response
    
    def get_conversation_chain_answser(self, question,memory,
                                    max_token: int = 2000,
                                    temperature: float = 0.01,
                                    top_p: float = 0.1):
        """
        输入包含:
        1. 当前题目
        2. 提出的问题
        输入格式: ["question": question, "timu":timu]
        """
        self.llm.max_token = max_token
        self.llm.top_p = top_p
        self.llm.temperature = temperature

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
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
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
        question_generator_chain = LLMChain(llm=self.llm, prompt=condense_prompt)

        conversation_chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_documents_chain,
        question_generator=question_generator_chain,
        retriever=self.vectore.as_retriever(),
        memory=memory,
        )
        response = conversation_chain({"question": question})

        return response
