import os
import time
import json
import sys
import torch 
import subprocess

from models import LLM
from utils import get_args, log_exp
from prompts import get_prompt
from datasets import get_lamp_ids
from retriever import get_retr_data, get_retr_res, prepare_context

args = get_args()
dataset = args.dataset
if dataset.startswith("lamp"):
    dataset_name = "lamp"
    dataset_params = {
        "num": int(dataset.split("_")[1]),
        "split": dataset.split("_")[-1]
    }
elif dataset.startswith("amazon"):
    dataset_name = "amazon"
    dataset_params = {
        "year": int(dataset.split("_")[1]),
        "category": dataset.split("_")[-1] 
    }
method = args.method
k = args.k
retriever = args.retriever if k != 0 else None
MAX_NEW_TOKENS = 64
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = get_lamp_ids(**dataset_params)    

LLMs = ["GEMMA-2-2B", "LLAMA-3.1-8B"]
print(f"Running experiments for the {dataset} using {method} method")
sys.stdout.flush()
for model_name in LLMs:
    exp_name = f"{dataset}_{model_name}_{method}_K{k}"
    pred_out_path = f"{pred_path}/{exp_name}.json"
    if os.path.exists(pred_out_path):
        with open(pred_out_path, "rb") as f:
             all_res = json.load(f)["golds"]
    else:
        all_res = []
    print(model_name) 
    queries, retr_texts, retr_gts = get_retr_data(dataset_name, **dataset_params)  
    if len(all_res) == len(queries):
        print("Experiment for this LLM is already concluded!")
        continue
    else:
        llm = LLM(model_name=model_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
        print(subprocess.run("gpustat"))  
        queries = queries[len(all_res):]
        retr_texts = retr_texts[len(all_res):]
        retr_gts = retr_gts[len(all_res):]
    if k != "0":
        retr_doc_idxs = get_retr_res(dataset, queries, retr_texts, retriever)
        retr_doc_idxs = retr_doc_idxs[len(all_res):]   

    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    sys.stdout.flush()

    all_context = prepare_context(dataset_name, retr_texts, retr_gts, retr_doc_idxs, k, **dataset_params)

    for i in range(len(queries)):

        query = queries[i]
        context = all_context[i]
        prompt = get_prompt(dataset_name, query, **dataset_params)
        context = llm.prepare_context(prompt, context)    
        prompt = get_prompt(dataset_name, query, **dataset_params, examples=context)

        start_bot_time = time.time()    
        res = llm.prompt_chatbot(prompt)
        end_bot_time = time.time()
        id = ids[i]["id"] if dataset_name == "lamp" else i

        cur_iter_res = {
            "id": id,
            "prompt": prompt[0]["content"],
            "output": res.strip(),
            "model_inf_time": round(end_bot_time - start_bot_time, 2) 
        }
        log_exp(cur_iter_res, exp_name)
        all_res.append({
            "id": id,
            "output": res.strip()
        })
        if (i+1)%500==0 or (i+1)==len(queries):
            print(i)
            with open(pred_out_path, "w") as f:
                task = f"LaMP_{dataset_params['num']}" if dataset_name == "lamp" else dataset          
                json.dump({
                    "task": task,
                    "golds": all_res
                }, f)
        sys.stdout.flush()
    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
    del llm
    llm = []
    torch.cuda.empty_cache()