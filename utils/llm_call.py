import time
from functools import wraps
import openai
from curl_cffi import requests

def retry_on_failure(max_retries=3, delay=1):
    """
    retry decorator
    
    Args:
        max_retries: maximum number of retries
        delay: retry interval (seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries += 1
                    if retries < max_retries:
                        print(f"model {self.model} request failed (attempt {retries}/{max_retries}): {str(e)}")
                        print(f"{delay} seconds later retry...")
                        time.sleep(delay)
            
            print(f"model {self.model} failed after {max_retries} retries: {str(last_error)}")
            raise last_error
            
        return wrapper
    return decorator

class CallLLM:
    def __init__(self, model:str = "Qwen/Qwen2.5-7B-Instruct", 
                 api_base:str = "http://xxx", 
                 api_key:str = "sk-xxx"):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        
    @retry_on_failure(max_retries=20, delay=4)
    def post_request(self, messages:list) -> tuple[str, int]:
        """
        send request and get answer, with retry mechanism
        """
        client = openai.OpenAI(base_url=self.api_base,api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content, response.usage.total_tokens
    
    @retry_on_failure(max_retries=20, delay=4)
    def post_reasoning_request(self, messages:list) -> tuple[str, int, str]:
        """
        send request and get answer, with retry mechanism
        """
        client = openai.OpenAI(base_url=self.api_base,api_key=self.api_key)
        if self.model == "o3-mini":
            
            response = client.chat.completions.create(
                model="o3-mini",
                reasoning_effort="high",
                messages=messages
            )
            return response.choices[0].message.content, response.usage.total_tokens, response.choices[0].message.content
        else:
            reasoning_content = ""
            answer_content = ""

            completion = client.chat.completions.create(
                model=self.model,  # use instance's model attribute
                messages=messages,
                stream=True,
                stream_options={
                    "include_usage": True
                }
            )
            usage_info = None
            for chunk in completion:
                
                if not chunk.choices:
                    usage_info = chunk.usage
                else:
                    delta = chunk.choices[0].delta
                    
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    
                    elif hasattr(delta, 'content') and delta.content is not None:
                        answer_content += delta.content

            
            # if answer_content:
            
            #     print(answer_content)
            
            # print(reasoning_content)
            
            # if usage_info:
                
                
                
                
            return answer_content, usage_info.total_tokens, reasoning_content
    