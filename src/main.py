import json
import os
from datetime import datetime

from cfg.cfg import get_nl_cfg
from utils import *
from RaLU import *
from extract import *

from evalplus.data.utils import stream_jsonl, write_jsonl

def run_math(args, metadata, record_dir):
    done = 0
    error = []
    if os.path.exists(os.path.join(record_dir, "token_usage.txt")):
        with open(os.path.join(record_dir, "token_usage.txt")) as rf:
            i_t, o_t = rf.read().split(",")
            i_t, o_t = int(i_t), int(o_t)
    else: i_t, o_t = 0, 0

    with open("../instructions/RaLU/math.txt") as sf, open("../instructions/RaLU/final_sys1.txt") as ff1, \
        open("../instructions/RaLU/final_sys1.txt") as ff2, open("../instructions/PoT_math_example.txt") as pf:
        SYSTEM_MSG, FINAL_SYS1, FINAL_SYS2, pot_example = sf.read().strip(), ff2.read().strip(), pf.read().strip()
    
    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec = json_obj["task_id"], json_obj["question"].strip()
        final_path = os.path.join(record_dir, task_id, "refine."+metadata['suffix'])
        if not os.path.exists(final_path):
            logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id} {'#'*15}")
        
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)
        ori_program_path = os.path.join(record_dir, task_id, f"ini.{metadata['suffix']}")
        
        if not os.path.exists(ori_program_path):
            logging.info("Write the initial program...")
            (program, program_ans), (i_token, o_token) = llm_write_execuable(
                    prompt_lst=[{"role": "user", "content": f"## Example\n{pot_example}"}, 
                                {"role": "user", "content": f"Question: {spec}\n#Python code, return ans."}],
                    system=f"You are an expert in writing Python to solve math questions. Wrap your program in a <code></code> block. " +\
                            "Python code, return ans. No more any test cases or other contents.",
                    model=args.model, max_valid_num=args.max_valid_num,
            )
            i_t += i_token; o_t += o_token
            with open(ori_program_path, "w") as wf:
                wf.write(program.strip())
                wf.write(f"\n\n##### {program_ans}\n")
        else:
            with open(ori_program_path) as rf:
                program = rf.read().strip()
        
        nl_cfg_path = os.path.join(record_dir, task_id, "cfg.json")
        if not os.path.exists(nl_cfg_path):
            logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id} {'#'*15}")
            logging.info("Generating CFG...")
            try:
                nl_cfg_lst = get_nl_cfg(ori_program_path, None)
            except:
                p_lst = [p.strip() for p in program.splitlines() if len(p.strip()) > 0]
                length = len(p_lst)
                third, remainder = length // 3, length % 3
                nl_cfg_lst = ["RUN [\n" + "\n".join(p_lst[:third])+"\n]",
                            "RUN [\n" + "\n".join(p_lst[third:2 * third + (remainder >= 1)])+"\n]",
                            "RUN [\n" + "\n".join(p_lst[2 * third + (remainder >= 1):])+"\n]"
                ]
        else:
            with open(nl_cfg_path) as rf:
                nl_cfg_lst = json.load(rf)

        final_path = os.path.join(record_dir, task_id, "refine."+metadata['suffix'])
        
        if not os.path.exists(final_path):
            logging.info("Aligning Logic Units based on CFG...")
            ralu = RaLU(ini_user_prompt=f"Question: {spec}\n\n",
                        system_msg=SYSTEM_MSG,
                        model=args.model, 
                        max_valid_num=args.max_valid_num
            )
            
            ralu.reason(nl_cfg_lst, max_branches=args.max_branches)
            json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))

            logging.info("Writing the final program...")

            final_program, final_answer = ralu.solving(spec=spec, system_lst=[FINAL_SYS1, FINAL_SYS2])
            with open(final_path, "w") as wf:
                wf.write(final_program)
                wf.write(f"\n\n##### {final_answer}\n")
            
            write_jsonl(os.path.join(record_dir, "answer.jsonl"), [{"task_id": task_id, "answer": final_answer}], append=True)
            i_t += ralu.token_consumption["input"]; o_t += ralu.token_consumption["output"]

        done += 1
        with open(os.path.join(record_dir, "token_usage.txt"), "w") as wf:
            wf.write(f"{i_t},{o_t}")
    
    logging.warning(f"{error}")
    return done, (i_t, o_t)
        

def run_code(args, metadata, record_dir):
    done = 0
    if os.path.exists(os.path.join(record_dir, "token_usage.txt")):
        with open(os.path.join(record_dir, "token_usage.txt")) as rf:
            i_t, o_t = rf.read().split(",")
            i_t, o_t = int(i_t), int(o_t)
    else: i_t, o_t = 0, 0

    with open("../instructions/RaLU/code.txt") as rf:
        SYSTEM_MSG = rf.read().strip()
    
    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec, entry_point = json_obj["task_id"], json_obj["prompt"].strip(), json_obj["entry_point"].strip()
        
        final_path = os.path.join(record_dir, task_id, "refine."+metadata['suffix'])
        if not os.path.exists(final_path):
            logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id}@{entry_point} {'#'*15}")
        
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)
        ori_program_path = os.path.join(record_dir, task_id, f"ini.{metadata['suffix']}")
        if not os.path.exists(ori_program_path):
            logging.info("Write the initial program...")
            program, (i_token, o_token) = llm_write_code(
                    prompt_lst=[{"role": "user", "content": f"## Specification\n{spec}"}],
                    system=f"You are an expert in {metadata['language']} coding. Wrap your program in a <code></code> block. No more any test cases or other contents.",
                    entry_point=entry_point,
                    language=metadata['language'],
                    model=args.model, 
                    max_valid_num=args.max_valid_num,
                )
            i_t += i_token; o_t += o_token
            with open(ori_program_path, "w") as wf:
                wf.write(program.strip())
        else:
            with open(ori_program_path) as rf:
                program = rf.read().strip()
        
        nl_cfg_path = os.path.join(record_dir, task_id, "cfg.json")
        if not os.path.exists(nl_cfg_path):
            logging.info("Generating CFG...")
            nl_cfg_lst = get_nl_cfg(ori_program_path, entry_point)
        else:
            with open(nl_cfg_path) as rf:
                nl_cfg_lst = json.load(rf)
        
        final_path = os.path.join(record_dir, task_id, "refine."+metadata['suffix'])
        if not os.path.exists(final_path):
            logging.info("Self-refining based on CFG...")
            
            ralu = RaLU(ini_user_prompt=f"## Specification\n{clean_spec(spec)}", system_msg=SYSTEM_MSG, model=args.model, max_valid_num=args.max_valid_num)
            ralu.reason(nl_cfg_lst, max_branches=args.max_branches)
            json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))
            logging.info("Writing the final program...")
            

            final_program = ralu.coding(spec=spec, entry_point=entry_point, language=metadata['language'], 
                system=f"You are an expert in {metadata['language']} coding. Your task is to write a correct program based on the given reasoning path. " + \
                    f"Show your thinking process explictly. Your program can contain several functions but make sure the main entry point is {entry_point}.\n" + \
                    "Example Response:\n<code>\ndef add_nums(a, b):\n    return a+b\n</code>\nAnalysis: Using operation `+` to implement the requirement of returning the sum of two numbers.\n" + \
                    "**Strict Requirement**: Ensure to return the complete program wraped in a <code></code> block! No test cases!"
            )
            i_t += ralu.token_consumption["input"]; o_t += ralu.token_consumption["output"]
            with open(final_path, "w") as wf:
                wf.write(final_program)
        
        done += 1
        with open(os.path.join(record_dir, "token_usage.txt"), "w") as wf:
            wf.write(f"{i_t},{o_t}")
    return done, (i_t, o_t)
    

if __name__ == "__main__":
    
    args = get_args()
    metadata = get_metadata(args)
    
    record_dir = args.record_dir if args.record_dir is not None else f"../records/{args.model}_" + datetime.now().strftime("%y%m%d_%H%M")
    logging_activate(record_dir)

    if args.dataset in ["HumanEvalPlus", "MbppPlus"]:
        done, (i_t, o_t) = run_code(args, metadata, record_dir)
    elif args.dataset in ["GSM8K", "MATH"]:
        done, (i_t, o_t) = run_math(args, metadata, record_dir)
    else:
        NotImplementedError

    logging.info(f"Using {i_t} input tokens and {o_t} output tokens in total | {round(i_t/done), round(o_t/done)} on average")
    
    if done == metadata['total_num'] and args.dataset in ["HumanEvalPlus", "MbppPlus"]:
        from prepare import *
        validate(generate_json_samples(args.dataset, record_dir, "refine."+metadata['suffix']), args.dataset)
        validate(generate_json_samples(args.dataset, record_dir, "ini."+metadata['suffix']), args.dataset)
