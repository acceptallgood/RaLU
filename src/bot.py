import numpy as np
import json
from retry import retry
from openai import OpenAI

class ConnectionError(Exception):
    pass

class LLMBot:
    def __init__(self, api_key="<your key>", model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="<local url>")
            
    @retry((ConnectionError, json.decoder.JSONDecodeError), tries=5, delay=1)
    def prompt_call(self, prompt_lst, system, confidence=False):
    
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + prompt_lst,
            logprobs=True,
            temperature=0.7,
            frequency_penalty=0.3,
        )
        self.i_token, self.o_token = response.usage.prompt_tokens, response.usage.completion_tokens
        if confidence:
            return response.choices[0].message.content, np.mean(np.round(np.exp(
                [log_prob.logprob for log_prob in response.choices[0].logprobs.content])* 100 , 2))
        else:
            return response.choices[0].message.content
