import os
import time
import json
import sys
import torch 
import subprocess
import copy

from models import LLM
from utils import get_args, get_k
from prompts import prepare_res_prompt, prepare_improvement_prompt
from exp_datasets import LampDataset, AmazonDataset
from feature_processor import FeatureProcessor
from retriever import Retriever

args = get_args()
if args.dataset.startswith("lamp"):
    dataset_name = "lamp"
    num = int(args.dataset.split("_")[1])
    split = args.dataset.split("_")[-1]
    dataset = LampDataset(num, split)
elif args.dataset.startswith("amazon"):
    dataset_name = "amazon"
    year = int(args.dataset.split("_")[-1])
    category = "_".join(args.dataset.split("_")[1:-1])
    dataset = AmazonDataset(category, year)
else:
    raise Exception("Dataset not known!")

MAX_NEW_TOKENS = 64 if dataset.name == "lamp" else 128
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = dataset.get_ids()    

# LLMs = ["GPT-4o-mini", "GEMMA-2-9B", "GEMMA-2-27B"]
LLMs = ["OPENCHAT-3.5", "GEMMA-2-9B", "GEMMA-2-27B"]
queries, retr_texts, retr_gts = dataset.get_retr_data() 
if not args.k:
    k = get_k(retr_texts)
else:
    k = args.k

retriever = Retriever(dataset, args.retriever)
final_feature_list = []
all_context = retriever.get_context(queries, retr_texts, retr_gts, k) 

if args.features:
    feature_processor = FeatureProcessor()
    all_features = feature_processor.get_all_features(args.dataset, args.features, retr_texts, retr_gts)
    prepared_features = feature_processor.prepare_features(all_features, args.features)
    final_feature_list = args.features
else:
    features = None

if args.counter_examples:
    _, query_retr_res = retriever.get_retrieval_results(queries, queries)
    final_feature_list.append(f"CE({args.counter_examples})")

print(f"Running experiments for {args.dataset} with Features: {final_feature_list}, Retriever: {args.retriever}, Two-step Gen: {args.two_step}, and K: {k}")
sys.stdout.flush()
for model_name in LLMs:
    if args.two_step:
        exp_name = f"{args.dataset}_{model_name}_{args.two_step}_{final_feature_list}_{args.retriever}_K({k}))"
    else:
        exp_name = f"{args.dataset}_{model_name}_{final_feature_list}_{args.retriever}_K({k}))"
    pred_out_path = f"{pred_path}/{exp_name}.json"
    if os.path.exists(pred_out_path):
        with open(pred_out_path, "rb") as f:
             all_res = json.load(f)["golds"]
    else:
        all_res = []
    print(model_name) 
    
    if len(all_res) == len(queries):
        print("Experiment for this LLM is already concluded!")
        continue

    llm = LLM(model_name=model_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    print(subprocess.run("gpustat")) 
    sys.stdout.flush() 
    cont_idx = copy.copy(len(all_res))

    for _ in range(len(queries) - len(all_res)):
        
        query = queries[cont_idx]       
        if dataset_name == "amazon":
            query_rating, _ = dataset.get_ratings(cont_idx) 
            query = f"{query}\nRating:\n{query_rating}"
        context = all_context[cont_idx]            
        if args.features:
            features = prepared_features[cont_idx]
        
        if args.counter_examples:
            _, retr_gt_name, retr_prompt_name = dataset.get_var_names()
            ce_idxs = query_retr_res[cont_idx][-args.counter_examples:]
            ce_examples = []
            for ce_i, ce in enumerate(ce_idxs):
                ce_example = []
                max_range = len(retr_texts[ce]) if k//4 > len(retr_texts[ce]) else k//4
                for j in range(max_range):
                    if retr_gt_name:
                        ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}\n{retr_gt_name.capitalize()}:\n{retr_gts[ce][j]}")
                    else:
                        ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}")
                ce_examples.append(ce_example)
        else:
            ce_examples = None

        start_bot_time = time.time() 

        prompt = prepare_res_prompt(dataset, query, llm, examples=context, features=features, counter_examples=ce_examples)
        prompt = [{"role": "user", "content": prompt}]
        res = llm.prompt_chatbot(prompt)

        if args.two_step:
            improvement_prompt = prepare_improvement_prompt(dataset, query, llm, context, features, res)
            improvement_prompt = [{"role": "user", "content": improvement_prompt}]
            res = llm.prompt_chatbot(improvement_prompt)

        # print(f"Pred: {res}")
        id = ids[cont_idx] if dataset_name == "lamp" else cont_idx
        # print(f"GT: {gts[cont_idx]}")
        end_bot_time = time.time()
        all_res.append({
                "id": id,
                "output": res.strip(),
                "prompt": prompt[0]["content"],
                "model_inf_time": round(end_bot_time - start_bot_time, 2), 
        })

        if (cont_idx+1)%500==0 or (cont_idx+1)==len(queries):
            print(cont_idx)
            with open(pred_out_path, "w") as f:
                task = f"LaMP_{num}" if dataset_name == "lamp" else args.dataset          
                json.dump({
                    "task": task,
                    "golds": all_res
                }, f)
        sys.stdout.flush()
        cont_idx += 1 
    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
    del llm
    llm = []
    torch.cuda.empty_cache()