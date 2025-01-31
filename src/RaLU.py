from bot import LLMBot
from uuid import uuid4
from collections import defaultdict
from copy import deepcopy
import re
from extract import *
from utils import *
import logging


def clean_spec(spec):
    func_define = ""
    if "def" in spec:
        for l in spec.splitlines():
            if "def" in l: 
                func_define = "\n" + l.replace("def", "Format of Target Function:").strip().strip(":") 
                break

    if spec.count('\"\"\"') == 2:
        spec = "\n".join([m.strip() for m in re.findall(r'"""(.*?)"""', spec, re.DOTALL)])
    
    for s in ["Write a function to", "Write a python function to", "Write a function that"]:
        if s in spec:
            spec = spec.replace(s, "").strip()
    spec += func_define
    return spec

def check_unit(unit, response):
    result = ""
    for c in response:
        if c.isalpha():
            result += c
        if result.lower() == "ok":
            return True
    block = extract_block(response)
    if block is None: return False
    return extract_core_chars(block) == extract_core_chars(unit)


class RaLU:
    def __init__(self, ini_user_prompt, system_msg=None, model="gpt-4o", max_valid_num=10):
        self.model = model
        self.max_valid_num = max_valid_num

        self.bot = LLMBot(model=model)
        self.current_id = uuid4().hex[:10]
        self.keep_state = defaultdict(list)
        self.system_msg = system_msg
        self.keep_state[self.current_id].append({"role": "user", "content": ini_user_prompt})

        self.token_consumption = {"input": 0, "output": 0}

    def branch_off(self):
        new_id = uuid4().hex[:10]
        self.keep_state[new_id] = deepcopy(self.keep_state[self.current_id])[:-2] #remove human's query and ai's response
        self.current_id = new_id
    
    def call_llm_without_update_state(self, prompt, confidence=True, sys=None):
        return self.bot.prompt_call(
            self.keep_state[self.current_id] + [{"role": "user", "content": prompt}], 
            system=self.system_msg if sys is None else sys, 
            confidence=confidence
        )

    def update_state(self, cid, prompt, response):
        self.keep_state[cid] += [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        logging.info(f"* State Update @ {cid}\n")

    
    def get_valid_response(self, unit, prompt):
        for vi in range(self.max_valid_num):
            logging.info(f"{self.current_id} | <Human>: {prompt}")
            response, score = self.call_llm_without_update_state(prompt)
            self.token_consumption["input"] += self.bot.i_token; self.token_consumption["output"] += self.bot.o_token
            logging.info(f"{self.current_id} | <AI> @ {score}: {response}")

            unit_correct_flag = check_unit(unit, response)
            
            if not unit_correct_flag:
                corrected_unit = extract_block(response)
                logging.info(f"{self.current_id} | <Env>: Extract corrected unit from AI's {vi}-th repeated response:\n{corrected_unit}")
                if corrected_unit is not None: return (False, corrected_unit, response, score)
            elif score > 51 or score < 0: 
                logging.info(f"{self.current_id} | <Env>: Correctness of this unit is checked by AI Judger")
                return (True, None, response, score)
    
    def most_confident_branch(self, branch_scores):
        if max(branch_scores.values()) > 0:
            logging.info(f"Cannot get a correct unit. Choose from scores of [" + \
                                ",".join([str(v) for v in branch_scores.values()]) + "]")
            return max(branch_scores, key=branch_scores.get)
        prompt = "## Candidates for Next Unit:\n"
        for bid in branch_scores:
            prompt += "<Unit>\n" + self.keep_state[bid][-2]["content"] + "\n</Unit>\n\n"

        logging.info(f"<Human>: {prompt}")
        for vi in range(self.max_valid_num):
            logging.info(f"{'^'*10} Judging the most confident branch /{vi+1}")
            response = self.bot.prompt_call(
                prompt_lst=[{"role": "user", "content": self.system_msg}] + self.keep_state[self.current_id][:-2] + [{"role": "user", "content": prompt}], 
                confidence=False,
                system="You are provided with a specification along with a series of logic units describing the control flow and discussions about the correctness of each unit. " + \
                f"Your goal is to select the correct next unit from {len(branch_scores)} controversial candidate units. Each candidate is warped with <Unit></Unit>. " + \
                "Pick the answer most reliable from them, again wrap it using <Unit></Unit>.\n **Note**: your response must repeat the existing answer, and should not change the listed answer."
            )
            
            self.token_consumption["input"] += self.bot.i_token; self.token_consumption["output"] += self.bot.o_token
            logging.info(f"<AI>: {response}")
            
            block = extract_block(response, "Unit")
            if block is not None:
                for bid in branch_scores:
                    if extract_core_chars(self.keep_state[bid][-2]["content"]) == extract_core_chars(block): 
                        logging.info(f"<Env>: Choose branch {bid}")
                        return bid


    def reason(self, nl_cfg_lst, max_branches=3): #nl_cfg_lst: a list of control graph described by natural language
        for unit_idx, unit in enumerate(nl_cfg_lst):
            corrected_unit = None
            branch_scores = {}
            unit_correct_flag = False

            prompt = f"Unit {unit_idx+1}: {unit}"
            logging.info(f"{'='*15} Unit {unit_idx+1} @ B-{self.current_id} {'='*15}")
            
            for branch_num in range(max_branches + 1):
                logging.info(f"{'-'*10} Switching {branch_num} branches for a self-checking Unit {unit_idx+1}")
                if branch_num > 0: 
                    prompt = f"Unit {unit_idx+1}: {corrected_unit}"
                    self.branch_off()
                if unit_idx == 0: prompt = "## Process\n" + prompt

                unit_correct_flag, corrected_unit, response, score = self.get_valid_response(unit, prompt)
                if unit_correct_flag:
                    self.update_state(self.current_id, prompt, response)
                    break
                
                assert corrected_unit is not None, response
                self.update_state(self.current_id, prompt, response)
                branch_scores[self.current_id] = score # Record and call llm again
            
            if not unit_correct_flag:
                assert len(branch_scores) == max_branches + 1
                self.current_id = self.most_confident_branch(branch_scores)
            
    
    def coding(self, system, spec, entry_point, language):
        states = self.keep_state[self.current_id][1:]
        units = []
        for i, state in enumerate(states):
            if state["role"] == "user":
                assert states[i+1]["role"] == "assistant", f"{state}\n\n{'-'*30}{states[i+1]}"
                units.append(state["content"])
                units.append(re.split(r"Analysis\s*:", states[i+1]["content"])[1].strip())
        
        prompt_lst = [{"role": "user", "content": "# Reasoning Path\n"+"\n\n".join(units)}, {"role": "user", "content": spec}]
        program, (i_token, o_token) = llm_write_code(
            system, prompt_lst, entry_point, language, model=self.model, max_valid_num=self.max_valid_num
        )
        self.token_consumption["input"] += i_token; self.token_consumption["output"] += o_token
        return program
    
    def solving(self, system_lst, spec):
        states = self.keep_state[self.current_id][1:]
        units = []
        for i, state in enumerate(states):
            if state["role"] == "user":
                assert states[i+1]["role"] == "assistant", f"{state}\n\n{'-'*30}{states[i+1]}"
                units.append(state["content"])
                units.append(re.split(r"Analysis\s*:", states[i+1]["content"])[1].strip())
        
        prompt_lst = [{"role": "user", "content": "# Reasoning Path\n"+"\n\n".join(units)}, 
                      {"role": "user", "content": "Question: " + spec}]
        
        (program, program_ans), (i_token, o_token) = llm_write_execuable(system_lst[0], prompt_lst, self.bot, self.max_valid_num)
        self.token_consumption["input"] += i_token; self.token_consumption["output"] += o_token
         
        if len(system_lst) > 1:
            prompt_lst += [{"role": "user", "content": f"## Assistant Program\n{program}\n\nExecution Result of `ans`: {program_ans}"}]
            (_, answer), (i_token, o_token) = llm_write_answer(system_lst[1], prompt_lst, self.bot, self.max_valid_num)
            self.token_consumption["input"] += i_token; self.token_consumption["output"] += o_token
            return program, answer
        else:
            return program, program_ans
            



        

