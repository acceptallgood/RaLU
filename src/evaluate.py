import os
from datetime import datetime

from utils import *
from extract import *
from evalplus.data.utils import stream_jsonl, write_jsonl

import timeout_decorator
from prepare import math_typing

@timeout_decorator.timeout(100)
def execute_code(code):
    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        logging.warning(f"<ENV> : Program Exec Error: {e}")
    return local_vars.get('ans', None)

@timeout_decorator.timeout(100)
def execute_file(file_path, p=False):
    local_vars = {}
    try:
        with open(file_path, 'r') as file:
            code = file.read()
            if p:
                print(code)
            exec(code, {}, local_vars)
        return local_vars.get('ans', None)
    except Exception as e:
        logging.warning(f"Exec Error {e}")
    return None

def cmp_lst(l1, l2, _type):
    if l1 is None or l2 is None: return False
    if len(l1) != len(l2): return False
    for i, j in zip(l1, l2):
        if _type == "str" and i != j: return False
        if _type == "int" and int(i) != int(j): return False
        if _type == "float" and abs(float(i)-float(j)) > 1e-4: return False
    return True

def str2tuple(_input, _type):
    if (_input.startswith("[") or _input.startswith("(")) and (_input.endswith("]") or _input.endswith(")")):
        _input = _input[1:-1]
    if _type == "str": return tuple([s for s in _input.split(",")])
    if _type == "int": return tuple([int(s) for s in _input.split(",")])
    if _type == "float": return tuple([float(s) for s in _input.split(",")])

def eval_math(record_dir, response_file):
    correct, total = 0, 0
    answers = {}
    invalid, incorrect = [], []
    for ans_item in stream_jsonl(os.path.join(record_dir, "answer.jsonl")):
        answers[ans_item["task_id"]] = ans_item["answer"]
    
    for json_obj in stream_jsonl(f'../dataset/MATH.jsonl'):
        task_id, gt_answer, _type = json_obj["task_id"], json_obj["gt_answer"], json_obj["ans_type"]
        test_answer = None
        
        total += 1
        if task_id in answers:
            test_answer = answers[task_id]
        
        if test_answer is None:
            if os.path.exists(os.path.join(record_dir, task_id, response_file)):
                with open(os.path.join(record_dir, task_id, response_file)) as rf:
                    program = rf.read()
                if "##### " in program: test_answer = program.split("##### ")[-1].strip()
                if test_answer == "None": test_answer = None
                if test_answer is None and response_file.endswith(".py"):
                    try: 
                        test_answer = execute_file(os.path.join(record_dir, task_id, response_file))
                    except:
                        pass
        
        if test_answer is None:
            invalid.append(task_id)
            print(f"{task_id} @ Invalid ### GT: {gt_answer} | TEST: {test_answer}")
            continue
        
        this_ans = None
        match = re.compile(r'\\boxed{(.*?)}|\\\((.*?)\\\)|\$(.*?)\$').search(str(test_answer))
        if match:
            for i in (1, 2, 3):
                if match.group(i) is not None:
                    test_answer = match.group(1)
        
        flag = False
        if _type in ["str", "latex"]:
            try:
                flag = (re.sub(r'\s+', '', str(gt_answer).lower()) == re.sub(r'\s+', '', str(test_answer).lower()))
            except Exception as e:
                pass

        elif "Tuple" in _type:
            ele_type = _type[:-1].split("Tuple[")
            try:
                this_ans = str2tuple(test_answer.lower().replace(" ", ""), ele_type)
                try:
                    flag = cmp_lst(gt_answer, this_ans, ele_type)
                except:
                    pass
            except Exception as e:
                pass
            
        
        elif _type == "int":
            try:
                flag = (int(gt_answer) == int(test_answer))    
            except Exception as e:
                try:
                    this_ans, _ = math_typing(task_id, test_answer)
                    if this_ans is not None:
                        flag = (int(gt_answer) == int(this_ans))
                except:
                    print(task_id, this_ans)
            
        
        elif _type == "float":
            try:
                flag = (abs(float(gt_answer) - float(test_answer)) < 1e-4 or round(float(gt_answer) ,3) == round(float(test_answer), 3)) 
            except (ValueError, TypeError) as e:
                try:
                    this_ans, _ = math_typing(task_id, test_answer)
                    flag = (abs(float(gt_answer) - float(this_ans)) < 1e-4 or round(float(gt_answer) ,3) == round(float(test_answer), 3)) 
                except:
                    print(task_id, this_ans)
            except Exception as e:
                pass
        
        else:
            NotImplementedError(_type)
        
        if not flag:
            incorrect.append(task_id)
            print(f"{'='*10} {task_id} @ {_type}: {gt_answer} | TEST: {test_answer}")
        correct += flag
    
    write_jsonl(os.path.join(record_dir, "answer.jsonl"), 
                    [{"task_id": task_id, "answer": test_answer} for task_id, test_answer in answers.items()])
    
    print(f"{'#'*40}\nCorrect: {correct} | InValid: {len(invalid)} | Total: {total} | Acc: {correct/total}")
    
    os.makedirs(os.path.join(record_dir, "result"), exist_ok=True)
    
    with open(os.path.join(record_dir, "result", "invalid.txt"), "w") as wf:
        wf.write("\n".join(invalid))
    
    with open(os.path.join(record_dir, "result", "incorrect.txt"), "w") as wf:
        wf.write("\n".join(incorrect))

def eval_gsm8k(record_dir, response_file):
    incorrect, invalid = [], []
    correct, total = 0, 0
    test_answers = {}
    if os.path.exists(os.path.join(record_dir, "answer.jsonl")):
        for json_obj in stream_jsonl(os.path.join(record_dir, "answer.jsonl")):
            test_answers[json_obj["task_id"]] = json_obj["answer"]

    for json_obj in stream_jsonl(f'../dataset/GSM8K.jsonl'):
        task_id, answer = json_obj["task_id"], json_obj["answer"].splitlines()[-1].strip().strip("#### ")
        if answer.isdigit() or (answer[0] == "-" and answer[1:].isdigit()): gt_answer = int(answer)
        elif "," in answer and answer.replace(",", "").isdigit(): gt_answer = int(answer.replace(",", ""))
        
        test_answer = None
        total += 1

        if task_id in test_answers:
            test_answer = test_answers[task_id]

        if test_answer is None or test_answer == "None":
            if os.path.exists(os.path.join(record_dir, task_id, response_file)):
                with open(os.path.join(record_dir, task_id, response_file)) as rf:
                    test_answer = rf.read().split("##### ")[-1].strip()
            if response_file.endswith(".py") and (test_answer == None or len(test_answer) == 0 or test_answer == "None"):
                try:
                    test_answer = execute_file(os.path.join(record_dir, task_id, response_file))
                except Exception as e:
                    logging.warning(f'{task_id} Timeout {e}')
        
        if test_answer is None:
            invalid.append(task_id)
            print(f"{task_id} @ Invalid ### GT: {gt_answer} | TEST: {test_answer}")
            continue

        if re.compile(r'^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$').match(str(test_answer)):
            hour = int(test_answer.split(':')[0]) % 12
            if hour == 0 and test_answer[0] != "0": test_answer = 12
        else: 
            try:
                test_answer = int(test_answer)
            except Exception as e:
                print(f"{'-'*20} {task_id}: {gt_answer} | TEST: {test_answer}")            
        
        if not (gt_answer == test_answer):
            incorrect.append(task_id)
            print(f"{task_id} @ False ### GT: {gt_answer} | TEST: {test_answer}")
            continue
        
        correct += 1
        
            

    write_jsonl(os.path.join(record_dir, "answer.jsonl"), 
                    [{"task_id": task_id, "answer": test_answer} for task_id, test_answer in test_answers.items()])
    print(f"{'#'*40}\nCorrect: {correct} | InValid: {len(invalid)} | Total: {total} | Acc: {correct/total}")
    os.makedirs(os.path.join(record_dir, "result"), exist_ok=True)
    
    with open(os.path.join(record_dir, "result", "invalid.txt"), "w") as wf:
        wf.write("\n".join(invalid))
    
    with open(os.path.join(record_dir, "result", "incorrect.txt"), "w") as wf:
        wf.write("\n".join(incorrect))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('record_dir')
    parser.add_argument('--dataset', default="GSM8K", choices=["GSM8K", "MATH"])
    parser.add_argument('--response_file', default="refine.py")
    args = parser.parse_args()

    if args.dataset == "GSM8K":
        eval_gsm8k(args.record_dir, args.response_file)
    elif args.dataset == "MATH":
        eval_math(args.record_dir, args.response_file)
    

