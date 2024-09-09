import os
import time
import json
import sys
import torch 
import subprocess

from models import LLM
from utils import get_args, get_k
from prompts import prepare_prompt
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

gts = dataset.get_gt()
retriever = Retriever(dataset, args.retriever)

MAX_NEW_TOKENS = 64
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = dataset.get_ids()    

LLMs = ["GPT-4o-mini", "LLAMA-3.1-8B", "GEMMA-2-9B"]
queries, retr_texts, retr_gts = dataset.get_retr_data() 
if not args.k:
    k = get_k(retr_texts)

all_context = retriever.get_context(queries, retr_texts, retr_gts, k) 
if args.features:
    feature_processor = FeatureProcessor()
    if args.feats_gt_only:
        all_features = feature_processor.get_features(dataset.tag, args.features, retr_gts)
    else:
        all_features = feature_processor.get_features(dataset.tag, args.features, retr_gts, retr_texts)

print(f"Running experiments for {args.dataset} with Features: {args.features} (Gen. step count: {args.step_gen}, Gt only: {args.feats_gt_only}), Retriever: {args.retriever}, and K: {k}")
sys.stdout.flush()
for model_name in LLMs:
    exp_name = f"{args.dataset}_{model_name}_{args.features}_{args.retriever}_{args.k}_{args.step_gen}_GT{args.feats_gt_only}"
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

    queries = queries[len(all_res):]
    retr_texts = retr_texts[len(all_res):]
    retr_gts = retr_gts[len(all_res):]
    gts = gts[len(all_res):]
    all_context = all_context[len(all_res):]
    if args.features:
        all_features = all_features[len(all_res):]

    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    print(subprocess.run("gpustat")) 
    sys.stdout.flush() 

    for i in range(len(queries)):

        query = queries[i]        
        context = all_context[i]            
        if args.features:
            if args.step_gen == 1:
                features = all_features[i]
            else:
                features = None
        else:
            if args.step_gen == 1:
                features = None
            else:
                raise Exception("For multi-step generation, features can't be empty!")

        prompt = prepare_prompt(dataset, query, llm, context, features)
        prompt = [{"role": "user", "content": prompt}]
        start_bot_time = time.time()    
        res = llm.prompt_chatbot(prompt)
        print(res)
        id = ids[i] if dataset_name == "lamp" else i
        if args.step_gen > 1:
            prompt = prepare_prompt(dataset, query, llm, context, features, res)
            prompt = [{"role": "user", "content": prompt}]
            print(prompt[0]["content"])
            print(res)
            res = llm.propmt_chatbot(prompt)

        end_bot_time = time.time()
        all_res.append({
                "id": id,
                "output": res.strip(),
                "prompt": prompt[0]["content"],
                "model_inf_time": round(end_bot_time - start_bot_time, 2), 
        })

        if (i+1)%500==0 or (i+1)==len(queries):
            print(i)
            with open(pred_out_path, "w") as f:
                task = f"LaMP_{num}" if dataset_name == "lamp" else args.dataset          
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