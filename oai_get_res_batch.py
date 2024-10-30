import os
import json
from openai import OpenAI

client = OpenAI()
batches = client.batches.list()
files = client.files.list()

for batch in batches:
    
    filename = [file.filename for file in files if file.id == batch.input_file_id]

    if filename:

        merged_res = []
        path_to_file = os.path.join("preds", filename[0])

        data = []
        with open(path_to_file, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        batch_res = client.files.content(batch.output_file_id).text
        batch_res = [json.loads(line) for line in batch_res.splitlines()]

        for sample in data:
            res = [res["response"]["body"]["choices"][0]["message"]["content"] for res in batch_res if res["custom_id"] == sample["custom_id"]]
            merged_res.append({
                "id": sample["custom_id"],
                "output": res[0].strip(),
                "prompt": sample["body"]["messages"][0]["content"],
                "model_inf_time": "n/a", 
        })
        with open(os.path.join("preds", f"{filename[0].split('.')[0]}.json"), "w") as f:
            task = f"LaMP_{filename[0].split('_')[1]}" if filename[0].startswith("lamp") else "_".join(filename[0].split("_")[:-3])
            json.dump({
                "task": task,
                "golds": merged_res
            }, f)