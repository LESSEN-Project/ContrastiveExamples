import json
import os
import re
import numpy as np

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

rand_k = 20
preds_dir = "preds"

out_gts = dataset.get_gt()
rand_samples = np.random.choice(range(len(out_gts)), rand_k, replace=False)

model_samples = {}
for file in os.listdir(preds_dir):
    if file.startswith(args.dataset):
        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]
        if len(preds) != len(out_gts):
            continue

        model_samples[file[len(args.dataset)+1:-5]] = [preds[idx] for idx in rand_samples]

for i, sample in enumerate(rand_samples):
    print(f"GT: {out_gts[sample]}")
    for model in model_samples.keys():
        print()
        print(model)
        print(model_samples[model][i])



        
