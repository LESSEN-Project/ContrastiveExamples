import os
import time
import json
import sys
import torch 
import subprocess

from models import LLM
from utils import get_args, log_exp
from prompts import get_prompt
from datasets import LampDataset, AmazonDataset
from retriever import Retriever
from personalize import get_context

args = get_args()
if args.dataset.startswith("lamp"):
    dataset_name = "lamp"
    num = int(args.dataset.split("_")[1])
    split = args.dataset.split("_")[-1]
    dataset = LampDataset(num, split)
elif args.dataset.startswith("amazon"):
    dataset_name = "amazon"
    year = int(args.dataset.split("_")[1])
    category = "_".join(args.dataset.split("_")[2:])
    dataset = AmazonDataset(category, year)
method = args.method
k = args.k
retriever_model = args.retriever if k != 0 else None
retriever = Retriever(retriever_model)

MAX_NEW_TOKENS = 64
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = dataset.get_ids()    

LLMs = ["GEMMA-2-2B", "LLAMA-3.1-8B"]
print(f"Running experiments for {args.dataset} using {method}")
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
    queries, retr_texts, retr_gts = dataset.get_retr_data()  
    if len(all_res) == len(queries):
        print("Experiment for this LLM is already concluded!")
        continue
    else:
        queries = queries[len(all_res):]
        retr_texts = retr_texts[len(all_res):]
        retr_gts = retr_gts[len(all_res):]
    if k != "0":
        _, retr_text_name, retr_gt_name = dataset.get_var_names()
        retr_doc_idxs = retriever.get_retrieval_results(args.dataset, queries, retr_texts, retriever)
        retr_doc_idxs = retr_doc_idxs[len(all_res):]  
        all_context = get_context(retr_texts, retr_gts, retr_doc_idxs, k, retr_text_name, retr_gt_name) 

    llm = LLM(model_name=model_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS}) 
    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    print(subprocess.run("gpustat")) 
    sys.stdout.flush() 

    for i in range(len(queries)):

        query = queries[i]
        prompt = get_prompt(dataset_name, query, num)
        if k != "0":
            context = all_context[i]
            context = llm.prepare_context(prompt, context)    
            prompt = get_prompt(dataset_name, query, num, examples=context)

        start_bot_time = time.time()    
        res = llm.prompt_chatbot(prompt)
        end_bot_time = time.time()
        id = ids[i] if dataset_name == "lamp" else i

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
                task = f"LaMP_{num}" if dataset_name == "lamp" else dataset          
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