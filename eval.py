import json
import pandas as pd
import os
from evaluate import load

from utils import get_args
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
cols = ["model", "method", "retriever"]

rouge = load("rouge")
cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])

for file in os.listdir(preds_dir):
    if file.startswith(args.dataset):
        retriever = file.split("_")[-1][:-5]
        features = file.split("_")[-2]
        if features != "None":
            features = ":".join(eval(features))
        model = file.split("_")[-3]

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(out_gts):
            continue

        print(model, retriever, features)
        res_dict = {    
            "model": model,
            "method": features,
            "retriever": retriever,
        }
        rouge_results = rouge.compute(predictions=preds, references=out_gts)
        all_res.append(res_dict | rouge_results)

df = pd.DataFrame(all_res)
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(os.path.join(evals_dir, f"eval_{args.dataset}.csv"), index=False, columns=cols)