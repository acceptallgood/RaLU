from .cfg_build import build_from_file
import sys
sys.path.append("..")
from utils import json_pretty_dump

def get_ori_cfg(program_file):
    cfg = build_from_file(program_file)
    graph = cfg.show(filepath=program_file.replace("ini.py", "cfg_graph"), show=False, calls=False)
    with open(program_file.replace("ini.py", "cfg.txt"), "w") as wf:
        wf.write(graph.source.strip())
    return cfg

def cfg_to_natural_language(cfg, entry_point):
    import math
    source = [s.strip() for s in "".join(cfg.traverse_print()).split("$") if len(s.strip()) > 0]
    if entry_point is not None:
        if source[0].startswith(f"RUN `[DEFINE FUNCTION] {entry_point}"):
            return source[1:]
    
    elif len(source) == 1:
        if len(source[0].splitlines()) > 9 and source[0].startswith("RUN ["):
            block = source[0].splitlines()[1:-1]
            div = math.ceil(len(source[0].splitlines()) / 3)
            return [
                "RUN [:\n" + "\n".join(block[: div]) + "\n]", 
                "RUN [:\n" + "\n".join(block[div: 2*div]) + "\n]", 
                "RUN [:\n" + "\n".join(block[2*div:]) + "\n]"
            ]

    return source

def get_nl_cfg(program_file, entry_point):
    source = cfg_to_natural_language(get_ori_cfg(program_file), entry_point)
    json_pretty_dump(source, program_file.replace("ini.py", "cfg.json"))
    return source


if __name__ == "__main__":
    import argparse
    import os
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('record_dir', help="dir that saves LLM-generated programs")
    args = parser.parse_args()

    attention = []

    if "Mbpp" in args.record_dir:
        with open('MbppPlus.jsonl', 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Java Parsing Error: {e}")
                
                task_id, spec, entry_point = json_obj["task_id"], json_obj["prompt"].strip(), json_obj["entry_point"].strip()
                assert os.path.exists(os.path.join(args.record_dir, task_id, "ini.py"))
                if len(attention) > 0 and task_id not in attention: continue
                for af in ["cfg.json", "cfg_graph.png"]:
                    if os.path.exists(os.path.join(args.record_dir, task_id, af)):
                        os.remove(os.path.join(args.record_dir, task_id, af))

                if not os.path.exists(os.path.join(args.record_dir, task_id, "cfg.json")):
                    print(f"Parsing {task_id} {entry_point}...")
                    cfg = get_ori_cfg(os.path.join(args.record_dir, task_id, "ini.py"))
                    if cfg.record:
                        with open("tmp/check.txt", "a+") as wf:
                            wf.write(task_id + "\n")
                    source = cfg_to_natural_language(cfg, entry_point)
                    json_pretty_dump(source, os.path.join(args.record_dir, task_id, "cfg.json"))
                else:
                    print(f"{task_id} cfg done")

    