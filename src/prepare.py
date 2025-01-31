import json
import os
from evalplus.data.utils import write_jsonl, stream_jsonl
import re

def generate_json_samples(dataset, record_dir, file_name):
    samples = []
    from evalplus.sanitize import script
    
    for json_obj in stream_jsonl(f'../dataset/{dataset}.jsonl'):    
        task_id = json_obj["task_id"]
        try:
            with open(os.path.join(record_dir, task_id, file_name)) as rf:
                samples.append(dict(task_id=task_id, solution=rf.read().strip()))
        except:
            print(os.path.join(record_dir, task_id, file_name))
            exit()

    samples_path = os.path.join(record_dir, file_name.replace(".py", ""), "samples.jsonl")
    os.makedirs(os.path.join(record_dir,  file_name.replace(".py", "")), exist_ok=True)
    write_jsonl(samples_path, samples)
    
    print("Sanitizing...")
    script(samples_path)
    return samples_path

def validate(samples_path, dataset):
    from evalplus.syncheck import syn_script
    print("#"*20, "Checking...")
    syn_script(samples_path, dataset.lower().replace("plus", ""))

    
def math_typing(task_id, ans):
    def insert_star(s):
        return re.compile(r'(\d)\s*(math\.)').sub(r'\1*\2', s)
    if ans is None: return None
    if ans.isdigit() or (ans[0] == "-" and ans[1:].isdigit()): 
        return int(ans), "int"
    if ans.isalpha(): return str(ans), "str"
    if "_" in ans and ans.replace("_", "").isdigit(): return str(ans), "str" 
    if re.compile('^\-?[0-9\s]+/[0-9\s]+$').match(str(ans)): 
        locals_dict = {}
        exec("result = " + ans, {}, locals_dict)
        return locals_dict['result'], "float"

    for char in ["i", "xyz", "qr", "b", "f"]:
        if re.compile(rf'^[{char}\+\-0-9\s\(\)\^=/.<]+$').match(ans.strip()): 
            return ans.replace(" ", ""), "str"
    
    if "begin{"+"pmatrix}" in ans or "\infty" in ans or "\cup" in ans or "cos" in ans: 
        return str(ans.replace(" ", "")), "latex"
    
    if "," in ans:
        ans = ans.replace(" ", "")
        if ans.startswith("(") and ans.endswith(")"):
            ans = ans[1:-1]
        if re.compile(r'^[\-0-9,\s]+$').match(ans):
            return tuple([int(i) for i in ans.split(",")]), "Tuple[int]"
        elif re.compile(r'^[a-zA-Z,\s]+$').match(ans):
            return tuple([str(i) for i in ans.split(",")]), "Tuple[str]"
        elif "." in ans and re.compile(r'^[\-.\d,]+$').match(ans):
            return tuple([float(i) for i in ans.split(",")]), "Tuple[float]"
        
    if "." in ans:
        if re.compile(r'^\d+(\.\d{1,4})?$').match(ans):
            return float(ans), "float"

    if ans.startswith("\(") and ans.endswith("\)"):
        ans = ans[2: -2]
    
    exp = ans.replace("\dfrac", "\frac")
    is_exp = False
    if "pi" in ans:
        exp = re.sub(r"(\\)?pi", "math.pi", ans)
        is_exp = True
    if "sqrt" in ans:
        pattern = re.compile(r'\\sqrt\{(-?\d+)\}')
        match = pattern.search(ans)
        if not match:
            print(task_id, "@", ans, "sqrt")
            return None, None
        is_exp = True
        exp = re.sub(r"\\sqrt{([^}]*)}", f"math.sqrt({int(match.group(1))})", exp)
    if "frac" in ans:
        pattern = re.compile(r'\\frac\{(.*?)\}\s*\{(.*?)\}')
        match = pattern.search(exp)
        if match:
            m, n = match.group(1), match.group(2)
            assert m.isdigit() or "math.sqrt" in m or "math.pi" in m,  f"{task_id} @ m: {m} | {ans}"
            assert n.isdigit() or "math.sqrt" in n or "math.pi" in n,  f"{task_id} @ n: {n} | {ans}"
            if ans.startswith("-\\frac"):
                exp = "-" + m +"/" + n
            elif ans.startswith("\\frac"):
                exp = m +"/" + n
            else:
                print(task_id, "@", ans, "frac")
                print(ans.startswith("\(\frac"), ans.endswith("\)"))
                return None, None
            is_exp = True

    if is_exp:
        exp = insert_star(exp)
        exp = "import math\nans = " + exp
        local_vars = {}
        try:
            exec(exp, {}, local_vars)
            ans = local_vars.get('ans', None)
            return float(ans), "float"
        except Exception as e:
            print(e)
            print(task_id, "@", ans, "|", exp)
            return None, None
    
    elif ans[0].isdigit():
        for pattern in [re.compile(r'(-?\d+)\}\s*\\[a-zA-Z]+\{'), re.compile(r'(-?\d+)\s*\\text\{[a-zA-Z\s]+\}')]:
            match = pattern.search(ans)
            if match:
                return int(match.group(1)), "int"
    
    return None, None