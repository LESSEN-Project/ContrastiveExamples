import json
import pandas as pd
import os
import re
from evaluate import load

from utils import get_args, parse_json
from exp_datasets import LampDataset, AmazonDataset

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

preds_dir = "preds"
evals_dir = "evals"
os.makedirs(preds_dir, exist_ok=True)
os.makedirs(evals_dir, exist_ok=True)

out_gts = dataset.get_gt()
all_res = []
models = []
cols = ["model", "features", "retriever", "k", "gen_step_count", "gt_only"]

rouge = load("rouge")
cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])

for file in os.listdir(preds_dir):
    if file.startswith(args.dataset):
        params = file[len(args.dataset)+1:-5].split("_")
        gt_only = re.findall(r'\((.*?)\)', params[5])[0]
        gen_step_count = re.findall(r'\((.*?)\)', params[4])[0]
        k = re.findall(r'\((.*?)\)', params[3])[0]
        retriever = params[2]
        features = params[1]
        if features != "None":
            features = ":".join(eval(features))
        model = params[0]

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]
            if int(gen_step_count) > 1:
                preds = [parse_json(p) for p in preds]

        if len(preds) != len(out_gts):
            continue

        print(model, retriever, features, gen_step_count, gt_only, k)
        res_dict = {    
            "model": model,
            "features": features,
            "retriever": retriever,
            "k": k,
            "gen_step_count": gen_step_count,
            "gt_only": gt_only,
        }
        rouge_results = rouge.compute(predictions=preds, references=out_gts)
        all_res.append(res_dict | rouge_results)

df = pd.DataFrame(all_res)
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(os.path.join(evals_dir, f"eval_{args.dataset}.csv"), index=False, columns=cols)