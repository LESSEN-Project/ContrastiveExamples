import os
import json

for file in os.listdir("preds"):

    if file.startswith("lamp") and file.split("_")[2] == "test":
        
        with open(os.path.join("preds", file), "r") as f:
            res = json.load(f)
        
        keys_to_keep = ["id", "output"]
        filtered_list = [{k: d[k] for k in keys_to_keep if k in d} for d in res["golds"]]

        res["golds"] = filtered_list
        os.makedirs("lamp_preds", exist_ok=True)
        with open(os.path.join("lamp_preds", file), "w") as f:
            json.dump(res, f)