import os
import time
import json

from openai import OpenAI

from models import LLM
from utils import get_args, get_k, oai_get_or_create_file
from prompts import prepare_res_prompt
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

MAX_NEW_TOKENS = 64 if dataset.name == "lamp" else 128
pred_path = "preds"
os.makedirs(pred_path, exist_ok=True)
if dataset_name == "lamp":
    ids = dataset.get_ids()    

model_name = "GPT-4o"

queries, retr_texts, retr_gts = dataset.get_retr_data() 
if not args.k:
    k = get_k(retr_texts)
else:
    k = args.k

retriever = Retriever(dataset, args.retriever)
final_feature_list = []
all_context = retriever.get_context(queries, retr_texts, retr_gts, k) 

if args.features:
    feature_processor = FeatureProcessor()
    all_features = feature_processor.get_all_features(args.dataset, args.features, retr_texts, retr_gts)
    prepared_features = feature_processor.prepare_features(all_features, args.features)
    final_feature_list = args.features
else:
    features = None

if args.counter_examples:
    _, query_retr_res = retriever.get_retrieval_results(queries, queries)
    final_feature_list.append(f"CE({args.counter_examples})")

print(f"Running batch experiments for {args.dataset} with Features: {final_feature_list}, Retriever: {args.retriever}, and K: {k}")

llm = LLM(model_name=model_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
exp_name = f"{args.dataset}_{model_name}_{final_feature_list}_{args.retriever}_K({k}))"

all_prompts = []

for i in range(len(queries)):
    
    query = queries[i]        
    context = all_context[i]            
    if args.features:
        features = prepared_features[i]
    
    if args.counter_examples:
        _, retr_gt_name, retr_prompt_name = dataset.get_var_names()
        ce_idxs = query_retr_res[i][-args.counter_examples:]
        ce_examples = []
        for ce_i, ce in enumerate(ce_idxs):
            ce_example = []
            max_range = len(retr_texts[ce]) if k//4 > len(retr_texts[ce]) else k//4
            for j in range(max_range):
                ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}\n{retr_gt_name.capitalize()}:\n{retr_gts[ce][j]}")
            ce_examples.append(ce_example)
    else:
        ce_examples = None

    id = ids[i] if dataset_name == "lamp" else i
    start_bot_time = time.time() 
    prompt = prepare_res_prompt(dataset, query, llm, examples=context, features=features, counter_examples=ce_examples)
    all_prompts.append({"custom_id": str(id), "method": "POST", "url": "/v1/chat/completions", 
                        "body": {"model": llm.repo_id, 
                                 "messages": [{"role": "user", "content":prompt}], "max_tokens": MAX_NEW_TOKENS}})
    
with open(os.path.join("preds", f"{exp_name}.jsonl"), "w") as file:
    for item in all_prompts:
        json_line = json.dumps(item)
        file.write(json_line + '\n')

client = OpenAI()
batch_input_file_id = oai_get_or_create_file(client, f"{exp_name}.jsonl")

client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)