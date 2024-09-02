import json
import pandas as pd
import os
from evaluate import load
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

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
out_gts = dataset.get_gt()
out_gts = [p["output"] for p in out_gts]
all_res = []
models = []
cols = ["model", "method", "retriever", "k"]
if dataset.task == "generation":
    rouge = load("rouge")
    cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
else:
    cols.append("acc")
    if num in [2, 3]:
        cols.append("f1")
    if num == 3:
        cols.extend(["mae", "rmse"])

for file in os.listdir(preds_dir):
    if file.startswith(args.dataset):
        retriever = file.split("_")[-1]
        k = file.split("_")[-2]
        method = file.split("_")[-3]
        model = file.split("_")[-4]

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(out_gts):
            continue

        print(model, retriever, method, k)
        res_dict = {
            "model": model,
            "method": method,
            "retriever": retriever,
            "k": k
        }
        if dataset.task == "generation":
            rouge_results = rouge.compute(predictions=preds, references=out_gts)
            all_res.append(res_dict | rouge_results)
        else:
            cor_pred = 0
            for i in range(len(out_gts)):
                if str(out_gts[i]) == str(preds[i]):
                    cor_pred += 1
            acc = cor_pred/len(out_gts)
            res_dict["acc"] = acc
            if num in [2, 3]:
                f1_macro = f1_score(out_gts, preds, average="macro")
                res_dict["f1"] = f1_macro
            if num == 3:
                mae = mean_absolute_error(list(map(float, out_gts)), list(map(float, preds)))
                rmse = mean_squared_error(list(map(float, out_gts)), list(map(float, preds)))
                res_dict["mae"] = mae
                res_dict["rmse"] = rmse
            all_res.append(res_dict)

df = pd.DataFrame(all_res)
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(f"eval_{args.dataset}_{args.method}.csv", index=False)