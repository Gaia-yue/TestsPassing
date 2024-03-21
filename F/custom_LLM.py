from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel,AutoConfig
import torch
import json
from abc import ABC

class ChatGLM3(LLM):
    max_token: int = 8192
    do_sample: bool = True
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()

    def _tool_history(self, prompt: str):
        ans = []

        tool_prompts = prompt.split(
            "You have access to the following tools:\n\n")[1].split("\n\nUse a json blob")[0].split("\n")
        tools_json = []

        for tool_desc in tool_prompts:
            name = tool_desc.split(":")[0]
            description = tool_desc.split(", args:")[0].split(":")[1].strip()
            parameters_str = tool_desc.split("args:")[1].strip()
            parameters_dict = ast.literal_eval(parameters_str)
            params_cleaned = {}
            for param, details in parameters_dict.items():
                params_cleaned[param] = {'description': details['description'], 'type': details['type']}

            tools_json.append({
                "name": name,
                "description": description,
                "parameters": params_cleaned
            })

        ans.append({
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following tools:",
            "tools": tools_json
        })

        dialog_parts = prompt.split("Human: ")
        for part in dialog_parts[1:]:
            if "\nAI: " in part:
                user_input, ai_response = part.split("\nAI: ")
                ai_response = ai_response.split("\n")[0]
            else:
                user_input = part
                ai_response = None

            ans.append({"role": "user", "content": user_input.strip()})
            if ai_response:
                ans.append({"role": "assistant", "content": ai_response.strip()})

        query = dialog_parts[-1].split("\n")[0]
        return ans, query

    def _extract_observation(self, prompt: str):
        return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
        self.history.append({
            "role": "observation",
            "content": return_json
        })
        return

    def _extract_tool(self):
        if len(self.history[-1]["metadata"]) > 0:
            metadata = self.history[-1]["metadata"]
            content = self.history[-1]["content"]

            lines = content.split('\n')
            for line in lines:
                if 'tool_call(' in line and ')' in line and self.has_search is False:
                    # 获取括号内的字符串
                    params_str = line.split('tool_call(')[-1].split(')')[0]

                    # 解析参数对
                    params_pairs = [param.split("=") for param in params_str.split(",") if "=" in param]
                    params = {pair[0].strip(): pair[1].strip().strip("'\"") for pair in params_pairs}
                    action_json = {
                        "action": metadata,
                        "action_input": params
                    }
                    self.has_search = True
                    print("*****Action*****")
                    print(action_json)
                    print("*****Answer*****")
                    return f"""
Action: 
```
{json.dumps(action_json, ensure_ascii=False)}
```"""
        final_answer_json = {
            "action": "Final Answer",
            "action_input": self.history[-1]["content"]
        }
        self.has_search = False
        return f"""
Action: 
```
{json.dumps(final_answer_json, ensure_ascii=False)}
```"""

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        if not self.has_search:
            self.history, query = self._tool_history(prompt)
        else:
            self._extract_observation(prompt)
            query = ""
        _, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        response = self._extract_tool()
        history.append((prompt, response))
        return response


class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p = 0.9
     history_len: int = 3

     def __init__(self,model_path):
         super().__init__()
         self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
         )
         self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto").eval()
         print("完成本地模型的加载")

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         messages = [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}
         ]
         text = self.tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
         generated_ids = self.model.generate(
             model_inputs.input_ids,
             max_new_tokens=512
         )
         generated_ids = [
             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
         ]

         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}
class ChatGLM(LLM):
    # 基于本地 llm 自定义 LLM 类
    max_token: int = 12000
    do_sample: bool = False
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, config=model_config, trust_remote_code=True, device_map="auto").eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """你是李文正的个人学习小助手.
        - 你是基于LLM的个人刷题助手, 你精通概念, 可以为用户生成题目.
        """
        messages = []
        response, history = self.model.chat(
        self.tokenizer,
        prompt,
        history=messages,
        do_sample=self.do_sample,
        max_length=self.max_token,
        temperature=self.temperature,
    )
        return response
        
    @property
    def _llm_type(self) -> str:
        return "CUSTOMLLM"
    
    def get_token_ids(self, text: str) -> List[int]:

        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
        return self.tokenizer.encode(text)