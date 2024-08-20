import argparse
import os
import pickle
import json
import urllib
import random
import re

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from contriever.src.contriever import Contriever

def log_exp(cur_iter, exp_name):
    os.makedirs("logs", exist_ok=True)
    result_path = os.path.join("logs", f"{exp_name}.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
    else:
        results = []
    with open(result_path, "w") as f:
        ids = [res["id"] for res in results]
        if cur_iter["id"] not in ids:
            results.append(cur_iter)
            json.dump(results, f)

def get_lamp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quant", default=None, type=str)
    parser.add_argument("-dn", "--dataset_num", default=5, type=int)
    parser.add_argument("-ds", "--dataset_split", default="train_dev", type=str)
    parser.add_argument("-k", "--k", default="3", type=str)
    parser.add_argument("-r", "--retriever", default="bm25", type=str)
    return parser.parse_args()

def get_lamp_dataset(dataset_num, mode="dev"):
    lamp_dataset_path = "datasets"
    os.makedirs(lamp_dataset_path, exist_ok=True)
    data_path = os.path.join(lamp_dataset_path, f"lamp_{dataset_num}_{mode}_data.json")
    gts = None
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        with urllib.request.urlopen(f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{dataset_num}/{mode}/{mode}_questions.json") as url:
            data = json.load(url)
        with open(data_path, "w") as f:
            json.dump(data, f)
    if mode != "test":
        gts_path = os.path.join(lamp_dataset_path, f"lamp_{dataset_num}_{mode}_gts.json")
        if os.path.exists(gts_path):
            with open(gts_path, "r") as f:
                gts = json.load(f)
        else:
            with urllib.request.urlopen(f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{dataset_num}/{mode}/{mode}_outputs.json") as url:
                gts = json.load(url)["golds"]
            with open(gts_path, "w") as f:
                json.dump(gts, f)
    return data, gts

def get_profvar_names(dataset_num):
    if dataset_num == 1:
        prof_gt_name = "title"
        prof_text_name = "abstract"
        prof_prompt_name = "abstract"
    elif dataset_num == 2:
        prof_gt_name = "category"
        prof_text_name = "text"
        prof_prompt_name = "article"        
    elif dataset_num == 3:
        prof_gt_name = "score"
        prof_text_name = "text"
        prof_prompt_name = "review"
    elif dataset_num == 4:
        prof_gt_name = "title"
        prof_text_name = "text"
        prof_prompt_name = "article"
    elif dataset_num == 5:
        prof_gt_name = "title"
        prof_text_name = "abstract"
        prof_prompt_name = "abstract"
    elif dataset_num == 7:
        prof_gt_name = None
        prof_text_name = "text"
        prof_prompt_name = "tweet"        

    return prof_text_name, prof_gt_name, prof_prompt_name

def create_retr_data(data, dataset_num=5):
    queries = []
    profile_text = []
    profile_gts = []
    prof_text_name, prof_gt_name, _  = get_profvar_names(dataset_num)
    for sample in data:
        if dataset_num in [3, 4, 5, 7]:
            text_idx = sample["input"].find(":") + 1
            queries.append(sample["input"][text_idx:].strip())
        elif dataset_num  == 1:
            queries.append(sample["input"].strip())
        elif dataset_num == 2:
            text_idx = sample["input"].find("article:") + 1
            queries.append(sample["input"][text_idx+len("article:"):].strip())
        if dataset_num != 7:
            profile_gts.append([p[prof_gt_name] for p in sample["profile"]])
        profile_text.append([p[prof_text_name] for p in sample["profile"]])
    return queries, profile_text, profile_gts

def retrieved_idx(prof_text, queries, dataset_num, dataset_split, model="bm25", device="cuda:0"):
    retr_path = f"retrievers/{dataset_num}/{dataset_split}"
    os.makedirs(retr_path, exist_ok=True)
    file_path = os.path.join(retr_path, f"{model}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            retr_doc_idxs = pickle.load(f)
    else:
        retr_doc_idxs = []
        if model == "bm25":
            for i in range(len(prof_text)):
                bm25 = BM25Okapi(prof_text[i])
                doc_scores = bm25.get_scores(queries[i])
                retr_doc_idxs.append(doc_scores.argsort()[::-1])
        elif model in ["contriever", "dpr"]:
            if model == "contriever":
                retr_model = Contriever.from_pretrained("facebook/contriever-msmarco") 
                tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            elif model == "dpr":
                retr_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
                tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")  
            retr_model.to(device).eval()
            with torch.no_grad():
                for i in range(len(prof_text)):
                    inp = prof_text[i]
                    inp.append(queries[i]) 
                    inputs = tokenizer(inp, padding=True, truncation=True, return_tensors="pt")
                    inputs.to(device)
                    embeddings = retr_model(**inputs)
                    if model == "dpr":
                        embeddings = embeddings.pooler_output
                    embeddings = embeddings.cpu()
                    sim_scores = np.dot(embeddings[-1:], embeddings[:-1].T)    
                    sorted_idxs = np.argsort(sim_scores).squeeze()[::-1]
                    retr_doc_idxs.append(sorted_idxs)
        else:
            raise Exception("Retriever not implemented!")     
        with open(file_path, "wb") as f:
            pickle.dump(retr_doc_idxs, f)
    return retr_doc_idxs

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