import json
from scipy.stats import ttest_rel
from evaluate import load

from exp_datasets import LampDataset

rouge = load("rouge")

baseline_4 = "preds/lamp_4_dev_GEMMA-2-27B_[]_contriever_K(50)).json"
baseline_5 = "preds/lamp_5_dev_GEMMA-2-27B_[]_contriever_K(7)).json"
baseline_7 = "preds/lamp_7_dev_GEMMA-2-27B_[]_contriever_K(50)).json"

baseline_res = {}

with open(baseline_4, "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    baseline_res["4"] = preds

with open(baseline_5, "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    baseline_res["5"] = preds

with open(baseline_7, "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    baseline_res["7"] = preds

best_method = {}

with open("preds/lamp_4_dev_GEMMA-2-27B_['WF', 'CE(3)']_contriever_K(50)).json", "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    best_method["4"] = preds

with open("preds/lamp_5_dev_GEMMA-2-27B_['DPF', 'CE(3)']_contriever_K(7)).json", "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    best_method["5"] = preds

with open("preds/lamp_7_dev_GEMMA-2-27B_['WF', 'DPF', 'CE(3)']_contriever_K(50)).json", "r") as f:
    preds = json.load(f)["golds"]
    preds = [p["output"] for p in preds]
    best_method["7"] = preds

datasets = ["4", "5", "7"]

for d in datasets:

    dataset = LampDataset(int(d), "dev")
    out_gts = dataset.get_gts()

    baseline_rouge = []
    best_rouge = []

    for i, elem in enumerate(out_gts):

        baseline_rouge.append(rouge.compute(predictions=[baseline_res[d][i]], references=[elem])["rougeL"])
        best_rouge.append(rouge.compute(predictions=[best_method[d][i]], references=[elem])["rougeL"])

    print(f"Dataset {d}:")

    t_stat, p_value = ttest_rel(baseline_rouge, best_rouge)

    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    alpha = 0.05 

    if p_value < alpha:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")