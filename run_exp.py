import os
import time
import json
import sys
import subprocess
import copy
import collections

import torch 

from models import LLM
from utils import get_args, get_k, parse_dataset
from prompts import prepare_res_prompt
from feature_processor import FeatureProcessor
from retriever import Retriever

args = get_args()
dataset = parse_dataset(args.dataset)

MAX_NEW_TOKENS = 64 if dataset.name == "lamp" else 128
MAX_NEW_TOKENS = MAX_NEW_TOKENS*8 if args.cot else MAX_NEW_TOKENS
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset.name == "lamp":
    ids = dataset.get_ids()    

LLMs = ["GEMMA-2-9B", "GEMMA-2-27B"]
# LLMs = ["GPT-4o-mini"]
queries, retr_texts, retr_gts = dataset.get_retr_data() 
if not args.top_k:
    k = get_k(retr_texts)
else:
    k = args.top_k

retriever = Retriever(dataset, args.retriever)
final_feature_list = []
all_context = retriever.get_context(queries, retr_texts, retr_gts, k) 

if args.features:
    feature_processor = FeatureProcessor()
    all_features = feature_processor.get_all_features(dataset.tag, args.features, retr_texts, retr_gts)
    prepared_features = feature_processor.prepare_features(all_features, args.features)
    final_feature_list = args.features
else:
    features = None

if args.counter_examples:
    ce_k = 3 if k == 50 else 1
    all_ce_examples = retriever.contrastive_retrieval(queries, retr_texts, retr_gts, args.counter_examples, ce_k)
    final_feature_list.append(f"CE({args.counter_examples})")

print(f"Running experiments for {dataset.tag} with Features: {final_feature_list}, Retriever: {args.retriever}, CoT: {args.cot}, Self Consistency: {args.num_consistency_samples}, and K: {k}")
sys.stdout.flush()

for model_name in LLMs:
    exp_name = f"{dataset.tag}_{model_name}_{final_feature_list}_{args.retriever}_CoT({args.cot})_SC({args.num_consistency_samples})_K({k}))"
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

    llm = LLM(model_name=model_name)
    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    print(subprocess.run("gpustat")) 
    sys.stdout.flush() 
    cont_idx = copy.copy(len(all_res))

    for _ in range(len(queries) - len(all_res)):
        
        query = queries[cont_idx]       
        if dataset.name == "amazon":
            query_rating, _ = dataset.get_ratings(cont_idx) 
            query = f"{query}\nRating:\n{query_rating}"
        context = all_context[cont_idx]            
        if args.features:
            features = prepared_features[cont_idx]
        
        if args.counter_examples:
            ce_examples = all_ce_examples[cont_idx]
        else:
            ce_examples = None

        start_bot_time = time.time() 
        prompt = prepare_res_prompt(dataset, query, llm, examples=context, features=features, counter_examples=ce_examples, use_cot_prompt=args.cot)
        prompt = [{"role": "user", "content": prompt}]

        res_list = []
        for i in range(args.num_consistency_samples):
            res = llm.prompt_chatbot(prompt, gen_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.8})
            res_list.append(res.strip())
        res_counter = collections.Counter(res_list)
        res = res_counter.most_common(1)[0][0]
        
        id = ids[cont_idx] if dataset.name == "lamp" else cont_idx
        end_bot_time = time.time()
        
        all_res.append({
                "id": id,
                "output": res,
                "prompt": prompt,
                "model_inf_time": round(end_bot_time - start_bot_time, 2), 
        })

        if (cont_idx+1)%500==0 or (cont_idx+1)==len(queries):
            print(cont_idx)
            with open(pred_out_path, "w") as f:
                task = f"LaMP_{dataset.num}" if dataset.name == "lamp" else dataset.tag          
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