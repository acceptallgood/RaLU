import argparse
def get_args(require_record_dir=False, require_baseline=False, require_ablation=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4o")
    parser.add_argument('--dataset', default="MbppPlus", choices=["MbppPlus", "HumanEvalPlus", "GSM8K", "MATH"])
    parser.add_argument('--record_dir', required=require_record_dir, default=None, help="dir that saves LLM-generated programs")
    parser.add_argument('--max_valid_num', default=1, type=int, help="allow the LLM to try multiple times in case of format errors")
    parser.add_argument('--max_branches', default=3, type=int, help="maximum number for rewind-and-correct")
    parser.add_argument('--ablation', type=str, choices=["line_by_line", "step_math", "step_code"], required=require_ablation)
    return parser.parse_args()

def get_metadata(args):
    language = {"MbppPlus": "Python", "HumanEvalPlus": "Python", "GSM8K": "Python", "MATH": "Python"}[args.dataset]
    suffix = {"Python": "py", "Java": "java"}[language]
    total_num = {"MbppPlus": 378, "HumanEvalPlus": 164, "GSM8K": 1319, "MATH": 700}[args.dataset]
    return {"language": language, "suffix": suffix, "total_num": total_num}

import json
def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
            separators=(",", ": "), ensure_ascii=False,)

import logging
import os
import shutil
def logging_activate(record_dir):
    os.makedirs(record_dir, exist_ok=True)
    if os.path.exists(os.path.join(record_dir, "run.log")): 
        for i in range(100):
            if not os.path.exists(os.path.join(record_dir, f"run-{i}.log")):
                shutil.copyfile(os.path.join(record_dir, "run.log"), os.path.join(record_dir, f"run-{i}.log"))
                print(i, "log")
                break
        os.remove(os.path.join(record_dir, "run.log"))
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(record_dir, "run.log")), logging.StreamHandler()],
    )