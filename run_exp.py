import os
import time
import json
import sys
import torch 
import subprocess

from models import LLM
from utils import get_args, log_exp, get_k
from exp_datasets import LampDataset, AmazonDataset
from feature_processor import FeatureProcessor
from retriever import Retriever
from personalize import Personalizer

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
retriever = Retriever(args.retriever)
personalizer = Personalizer(dataset, retriever)

MAX_NEW_TOKENS = 64
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = dataset.get_ids()    

LLMs = ["GPT-4o-mini", "LLAMA-3.1-8B", "GEMMA-2-9B"]
print(f"Running experiments for {args.dataset} using features: {args.features} and retriever: {args.retriever}")
sys.stdout.flush()

queries, retr_texts, retr_gts = dataset.get_retr_data() 
k = get_k(retr_texts)
print(f"K: {k}")
all_context = personalizer.get_context(queries, retr_texts, retr_gts, k) 

feature_processor = FeatureProcessor()
if args.feats_gt_only:
    features = feature_processor.get_features(dataset.tag, args.features, retr_gts)
else:
    features = feature_processor.get_features(dataset.tag, args.features, retr_gts, retr_texts)

for model_name in LLMs:
    exp_name = f"{args.dataset}_{model_name}_{','.join(args.features)}_{args.retriever}"
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
    if features:
        features = features[len(all_res):]

    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    print(subprocess.run("gpustat")) 
    sys.stdout.flush() 

    for i in range(len(queries)):

        query = queries[i]        
        context = all_context[i]
        print("".join(context))
        prompt = personalizer.prepare_prompt(args.method, query, llm, context)
        prompt = [{"role": "user", "content": prompt}]
        start_bot_time = time.time()    
        res = llm.prompt_chatbot(prompt)
        print(f"Pred: {res}")
        print(f"GT: {gts[i]}")
        print()
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