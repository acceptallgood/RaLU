import os
from datetime import datetime
from utils import *
from RaLU import *
from extract import *

from evalplus.data.utils import stream_jsonl, write_jsonl

def step_division_math(args, metadata, record_dir):
    with open("../instructions/RaLU/step_math.txt") as rf:
        SYSTEM_MSG = rf.read().strip()
    done = 0
    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec = json_obj["task_id"], json_obj["question"].strip()
        final_path = os.path.join(record_dir, task_id, args.refine_name+"."+metadata['suffix'])
        if os.path.exists(final_path):
            done += 1
            continue
        
        logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id} {'#'*15}")
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)

        model = LLMBot(model=args.model)
        steps = None
        for _ in range(args.max_valid_num):
            response = model.prompt_call(
                prompt_lst=[{"role": "user", "content": f"Question: {spec}"}],
                system="You are an expert in solving math questions. Your goal is to return the final answer to solve the given question and show your thinking process explicitly." + \
                        "Let's think it step by step. For example:\nQuestion: John has 10 apples. He gives away 4 and then receives 5 more. How many apples does he have?\n" + \
                        "Response:\n<Step>1: John starts with 10 apples.</Step>\n<Step>2: He gives away 4, so 10 - 4 = 6.</Step>\n<Step>3: He then receives 5 more apples, so 6 + 5 = 11.</Step>\n" + \
                        "<Answer>11</Answer>\n\n**Strict Requirement**: Wrap each step in a <Step></Step> block and wrap your final answer in a <Answer></Answer> block! ",
                confidence=False
            )
            logging.info(f"<AI>: {response}")
            steps = extract_block(response, "Step", True)
            if steps is not None:
                break
        assert steps is not None
        logging.info(f"<ENV> Extract {len(steps)} steps")
        ralu = RaLU(ini_user_prompt=f"Question: {spec}\n\n", system_msg=SYSTEM_MSG, model=args.model, max_valid_num=args.max_valid_num)
        ralu.reason(steps, max_branches=args.max_branches)
        json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))
        
        ans_type = "int" if 'ans_type' not in json_obj else json_obj['ans_type']
        system_lst = [f"You are an expert in writing Python to solve math questions. Your task is to write a correct Python program based on the given reasoning path to solve the math question by returning `ans`\n" + \
                    f"Show your thinking process explictly.\n\n**Strict Requirement**: Ensure to return the complete program wraped in a <code></code> block!",
                    
                    f"You are an expert in solving math questions. We have an assistant program based on the history reasoning path. Your task is to return the final answer to solve the given question, " +\
                    f"Based on the above reasoning path, the given program, and the `ans` calculated by the program. Show your thinking process explictly.\n" + \
                    f"**Strict Requirement**: Wrap your final answer in a <Answer></Answer> block! The answer should be in the format of {ans_type}."
        ]
        
        final_program, final_answer = ralu.solving(spec=spec, system_lst=system_lst)
        with open(final_path, "w") as wf:
            wf.write(final_program)
            wf.write(f"\n\n##### {final_answer}\n")
        write_jsonl(os.path.join(record_dir, "answer.jsonl"), [{"task_id": task_id, "answer": final_answer}], append=True)

        done += 1
    return done

def step_division_code(args, metadata, record_dir):
    with open("../instructions/RaLU/step_code.txt") as rf:
        SYSTEM_MSG = rf.read().strip()
    
    done = 0
    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec, entry_point = json_obj["task_id"], json_obj["prompt"].strip(), json_obj["entry_point"].strip()
        
        final_path = os.path.join(record_dir, task_id, args.refine_name+"."+metadata['suffix'])
        if os.path.exists(final_path):
            done += 1
            continue
        logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id}@{entry_point} {'#'*15}")
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)
        
        model = LLMBot(model=args.model)
        steps = None
        for _ in range(args.max_valid_num):
            response = model.prompt_call( 
                prompt_lst=[{"role": "user", "content": f"## Specification\n{clean_spec(spec)}\nEntry point: {entry_point}"}],
                system="You are an expert in solving programming questions. Your goal is to return the thinking steps to solve the given specification without programming. Show your thinking process explictly." + \
                        "Let's think it step by step. For example:\nQuestion: John has 10 apples. He gives away 4 and then receives 5 more. How many apples does he have?\n" + \
                        "Response:\n<Step>1: John starts with 10 apples.</Step>\n<Step>2: He gives away 4, so 10 - 4 = 6.</Step>\n<Step>3: He then receives 5 more apples, so 6 + 5 = 11.</Step>\n\n" + \
                        "**Strict Requirement**: Wrap each step in a <Step></Step> block! Do not add any code, test cases, or other contents!",
                confidence=False
            )
            logging.info(f"<AI>: {response}")
            steps = extract_block(response, wrap="Step", multiple=True)
            if steps is not None:
                break
        assert steps is not None
        
        ralu = RaLU(f"## Specification\n{clean_spec(spec)}", SYSTEM_MSG, args.model, max_valid_num=args.max_valid_num)
        ralu.reason(steps, max_branches=args.max_branches)
            
        final_program = ralu.coding(spec=spec, entry_point=entry_point, language=metadata['language'], 
            system=f"You are an expert in {metadata['language']} coding. Your task is to write a correct program to meet the specification based on the previous conversation. " + \
                    f"Show your thinking process explictly. Example Response:\n<code>\ndef add_nums(a, b):\n    return a+b\n</code>\n" + \
                    "Analysis: Using operation `+` to directly implement the requirement of returning the sum of two numbers." + \
                    f"Your program can contain several functions but make sure the main entry point is {entry_point}.\n" + \
                    "**Strict Requirement**: Ensure to return the complete function wraped in a <code></code> block! No test cases!"
        )
        json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))
        with open(final_path, "w") as wf:
            wf.write(final_program)
        done += 1
    return done

def line_by_line_math(args, metadata, record_dir):
    with open("../instructions/Ralu/line_by_line.txt") as rf:
        LINEBYLINE_SYSTEM_MSG = rf.read().strip()
    
    done = 0

    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec = json_obj["task_id"], json_obj["question"].strip()
        final_path = os.path.join(record_dir, task_id, args.refine_name+"."+metadata['suffix'])
        if os.path.exists(final_path):
            done += 1
            continue

        logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id} {'#'*15}")
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)
        with open(f"../save/{args.model}_{args.dataset.replace('Plus', '').lower()}/{task_id}/ini.{metadata['suffix']}") as rf:
            ori_program = rf.read()
        
        logging.info("Self-refining line by line...")

        ralu = RaLU(ini_user_prompt=f"Question: {spec}\n\n", system_msg=LINEBYLINE_SYSTEM_MSG, model=args.model,  max_valid_num=args.max_valid_num)
        ralu.reason([s.strip() for s in ori_program.splitlines() if len(s.strip()) > 0], max_branches=args.max_branches)

        json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))
    
        ans_type = "int" if 'ans_type' not in json_obj else json_obj['ans_type']
        system_lst = [f"You are an expert in writing Python to solve math questions. Your task is to write a correct Python program based on the given reasoning path to solve the math question by returning `ans`\n" + \
                    f"Show your thinking process explictly.\n\n**Strict Requirement**: Ensure to return the complete program wraped in a <code></code> block!",
                    f"You are an expert in solving math questions. We have an assistant program based on the history reasoning path. Your task is to return the final answer to solve the given question, " +\
                    f"Based on the above reasoning path, the given program, and the `ans` calculated by the program. Show your thinking process explictly.\n" + \
                    f"**Strict Requirement**: Wrap your final answer in a <Answer></Answer> block! The answer should be in the format of {ans_type}."
        ]
        
        final_program, final_answer = ralu.solving(spec=spec, system_lst=system_lst)
        with open(final_path, "w") as wf:
            wf.write(final_program)
            wf.write(f"\n\n##### {final_answer}\n")
        write_jsonl(os.path.join(record_dir, "answer.jsonl"), [{"task_id": task_id, "answer": final_answer}], append=True)

        done += 1
    return done
        

def line_by_line_code(args, metadata, record_dir):
    done = 0
    with open("../instructions/Ralu/line_by_line.txt") as rf:
        LINEBYLINE_SYSTEM_MSG = rf.read().strip()
    
    for json_obj in stream_jsonl(f'../dataset/{args.dataset}.jsonl'):
        task_id, spec, entry_point = json_obj["task_id"], json_obj["prompt"].strip(), json_obj["entry_point"].strip()
        ablation_path = os.path.join(record_dir, task_id, args.refine_name+"."+metadata['suffix'])
        
        if os.path.exists(ablation_path):
            done += 1
            continue
        logging.info(f"{'#'*15} Processing {done+1}/{metadata['total_num']}: {task_id}@{entry_point} {'#'*15}")
        
        os.makedirs(os.path.join(record_dir, task_id), exist_ok=True)
        assert os.path.exists(f"../save/{args.model}_{args.dataset.replace('Plus', '').lower()}/{task_id}/ini.{metadata['suffix']}")
        with open(f"../save/{args.model}_{args.dataset.replace('Plus', '').lower()}/{task_id}/ini.{metadata['suffix']}") as rf:
            ori_program = rf.read()
        
        logging.info("Self-refining line by line...")

        ralu = RaLU(ini_user_prompt=f"## Specification\n{clean_spec(spec)}\n\n",system_msg=LINEBYLINE_SYSTEM_MSG, model=args.model,  max_valid_num=args.max_valid_num)
        ralu.reason([s.strip() for s in ori_program.splitlines() if len(s.strip()) > 0], max_branches=args.max_branches)

        logging.info("Writing the final program...")
        
        refined_program = ralu.coding(spec=spec, entry_point=entry_point, language=metadata['language'], system=
                            "You are a helpful assistant in coding. There have been code lines of a program responding to the specification." + \
                            f"Based on the given lines and analysis, your task is to write a correct {metadata['language']} program to meet the specification." + \
                            "Wrap your program in a <code></code> block. For example:\n<code>\ndef add_nums(a, b):\n    return a+b\n</code>\n" + \
                            f"Your program can contain several functions but make sure the entry point is {entry_point}.\n" + \
                            "**Note**: Your response should follow the given conversation. Return your written program only! No more any other contents!"
        )
        
        json_pretty_dump(ralu.keep_state, os.path.join(record_dir, task_id, "branches.json"))
        with open(ablation_path, "w") as wf:
            wf.write(refined_program)
        
        done += 1
    return done

if __name__ == "__main__":
    
    args = get_args(require_ablation=True)
    metadata = get_metadata(args)
    
    record_dir = args.record_dir if args.record_dir is not None else f"../ablation/{args.ablation}/{args.model}_" + datetime.now().strftime("%y%m%d_%H%M")
    logging_activate(record_dir)
    
    if args.ablation == "line_by_line":
        if args.dataset in ["HumanEvalPlus", "MbppPlus"]:
            done = line_by_line_code(args, metadata, record_dir)
        else:
            done = line_by_line_math(args, metadata, record_dir)
    elif args.ablation == "step_math":
        done = step_division_math(args, metadata, record_dir)
    elif args.ablation == "step_code":
        done = step_division_code(args, metadata, record_dir)
    
    print("Finished")
    if done == metadata['total_num'] and args.dataset in ["HumanEvalPlus", "MbppPlus"]:
        from prepare import *
        validate(generate_json_samples(args.dataset, args.record_dir, "refine."+metadata['suffix']), args.dataset)
