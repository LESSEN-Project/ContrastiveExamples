import argparse
import os
import json
import random
import re
import numpy as np
import time

def log_exp(cur_iter, exp_name):
    os.makedirs("logs", exist_ok=True)
    result_path = os.path.join("logs", f"{exp_name}_{time.time()}.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
    else:
        results = []
    with open(result_path, "w") as f:
        results.append(cur_iter)
        json.dump(results, f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="lamp_5_dev", type=str)
    parser.add_argument("-k", "--k", default=None, type=int)
    parser.add_argument('-f', '--features', nargs='+', type=str, default=None)
    parser.add_argument("-r", "--retriever", default="contriever", type=str)
    parser.add_argument("-ce", "--counter_examples", default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def list_files_in_directory(root_dir):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def lamp_output_formatter(output, dataset_num):
    if dataset_num == 3:
        substring = "0"
        for c in output:
            try:
                if c.isdigit():
                    if 0 < int(c) and int(c) < 6:
                        substring = c
                        break
            except Exception as e:
                print(e)
                substring = "0"
    elif dataset_num == 5:
        dq_match = re.search(r'"([^"]*)"', output)
        if dq_match:
            substring = dq_match.group(0)
        else:
            substring = output
        substring = substring.strip('"')
        title_index = substring.find("Title:")
        if title_index != -1:
            substring = substring[title_index + len("Title:"):]
        substring = substring.strip()
        ex_index = substring.find("</EXAMPLES>")
        if ex_index != -1:
            substring = substring[:ex_index]
        substring = substring.strip()
        angle_b_index = substring.find("</s>")
        if angle_b_index != -1:
            substring = substring[:angle_b_index]
        substring.strip()
        note_index = substring.find("Note:")
        if note_index != -1:
            substring = substring[:note_index]
        substring = substring.strip()
        nl_index = substring.find("\n")
        if nl_index != -1:
            substring = substring[nl_index:]
    else:
        substring = output
    return substring.strip()

def shuffle_lists(list1, list2):
   zipped_list = list(zip(list1, list2))
   random.shuffle(zipped_list)
   list1_shuffled, list2_shuffled = zip(*zipped_list)
   list1_shuffled = list(list1_shuffled)
   list2_shuffled = list(list2_shuffled)
   return list1_shuffled, list2_shuffled

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x/e_x.sum()).tolist()

def get_k(retr_texts):
    mean = []
    for retr_text in retr_texts:
        mean.append(np.mean([len(text.split(" ")) for text in retr_text]))
    mean = np.mean(mean)
    if mean < 50:
        return 50
    else:
        return 7
    
def parse_json(output):
    try:
        idx = output.find("{")
        if idx != 0:
            output = output[idx:]
            if output.endswith("```"):
                output = output[:-3]
        output = json.loads(output, strict=False)["Title"]
    except Exception as e:
        try:
            match = re.search(r'"Title":\s*(.+)$', output, re.MULTILINE)
            if match:
                return match.group(1).strip().rstrip(',').strip()
            else:
                match = re.search(r'"title":\s*(.+)$', output, re.MULTILINE)
                if match:
                    return match.group(1).strip().rstrip(',').strip()
        except Exception as e:
            print(output)
            print(e)
    return output