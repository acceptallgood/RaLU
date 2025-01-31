from bot import LLMBot
import re
import logging

def extract_block(response, wrap="Fix", multiple=False):
    for w in [wrap, wrap.lower(), wrap.upper()]:
        match = re.search(rf'<\s*{w}\s*>(.*?)<\s*/\s*{w}\s*>', response, re.DOTALL)
        if match:
            if not multiple: return match.group(1).strip()
            else: return re.findall(rf'<\s*{w}\s*>(.*?)<\s*/\s*{w}\s*>', response, re.DOTALL)
    return None


def extract_core_chars(s):
    return re.sub(r'[^a-zA-Z0-9()+\-*/=<>!&|^%]', '', s.lower())

def check_program(program, entry_point, language):
    if language == "Python":
        return (re.search(rf"def {entry_point}\s*\(", program) is not None)
    else:
        NotImplementedError

def extract_program(response, entry_point, language="Python"):
    respose = response.replace("`<code>`", "")
    match = extract_block(response, "code", False)
    if match:
        program = match
    
    elif "```" in response:
        matches = re.findall(r"\n```.*?\n(.*?)\n```\n", response, re.DOTALL)
        if matches:
            program = matches[-1].strip()
        else:
            candidate = response.splitlines()
            if candidate[0].startswith("```"): candidate = candidate[1:]
            if candidate[-1].startswith("```"): candidate = candidate[:-1]
            program = "\n".join(candidate)

    elif response.strip().startswith(f"def {entry_point}"):
        candidate = []
        for line in response.splitlines():
            if line.startswith("def") or line.startswith("\t") or line.startswith("    "):
                candidate.append(line.rstrip())
        program = "\n".join(candidate)
        print("="*15, "Warning!", "="*15)
        print(program)
        print("="*40)

    else: return None
    
    if entry_point is None:
        return program
    
    if entry_point not in program and entry_point.lower() in program:
        program = program.replace(entry_point.lower(), entry_point)
    if check_program(program, entry_point, language):
        return program
    else:
        program = re.sub(r"def\s[a-zA-Z_]+\(", f"def {entry_point} (", program)
        if check_program(program, entry_point, language):
            return program

    return None

def llm_write_code(system, prompt_lst, entry_point, language, model="gpt-4o", max_valid_num=10):
    model = LLMBot(model=model) if isinstance(model, str) else model

    for _ in range(max_valid_num):
        response = model.prompt_call( prompt_lst, system, confidence=False)
        logging.info(f"<AI>: {response}")
        program = extract_program(response, entry_point, language)
        if program is not None: return program, (model.i_token, model.o_token)
    
    raise ValueError("No valid program!")

def llm_write_execuable(system, prompt_lst, model="gpt-4o", max_valid_num=10):
    from evaluate import execute_code
    model = LLMBot(model=model) if isinstance(model, str) else model
    
    for _ in range(max_valid_num):
        response = model.prompt_call( prompt_lst, system, confidence=False)
        logging.info(f"<AI>: {response}")
        program = extract_program(response, None, "Python")
        if program is not None:
            try:
                program_ans = execute_code(program)
                return (program, program_ans), (model.i_token, model.o_token)
            except Exception as e:
                logging.warning(f"<ENV> Timeout Exec: {e}")
                continue
    
    raise ValueError("No executable program!")
        

def llm_write_answer(system, prompt_lst, model="gpt-4o", max_valid_num=10):
    model = LLMBot(model=model) if isinstance(model, str) else model
    for _ in range(max_valid_num):
        response = model.prompt_call( prompt_lst, system, confidence=False)
        logging.info(f"<AI>: {response}")
        ans = extract_block(response, "Answer")
        if ans is None and "the answer is" in response.lower():
            ans = response.lower().split("the answer is")[-1].strip()
        logging.info(f"<ENV>: Extract Answer: {ans}")
        return (response, ans), (model.i_token, model.o_token)

    return (response, None), (model.i_token, model.o_token)

    
