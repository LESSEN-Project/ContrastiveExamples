import os
import json
import torch
import numpy as np

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from contriever.src.contriever import Contriever
from datasets import get_lamp_var_names, get_dataset, get_lamp_var_names
from utils import shuffle_lists

def prepare_context(dataset_name, retr_texts, retr_gts, retr_doc_idxs, k, **dataset_params):
    if dataset_name == "lamp":
        return _prepare_lamp_context(retr_texts, retr_gts, retr_doc_idxs, k, dataset_params["num"])

def get_retr_data(dataset_name, **dataset_params):
    data = get_dataset(dataset_name, **dataset_params)
    if dataset_name == "lamp":
        return _get_lamp_retr_data(data, dataset_params["num"])
    elif dataset_name == "amazon":
        return _get_amazon_retr_data(data)
    
def get_retr_res(dataset, queries, retr_texts, retriever):
    retr_path = f"retrieval_res"
    os.makedirs(retr_path, exist_ok=True)
    file_path = os.path.join(retr_path, f"{dataset}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return _find_retr_res(queries, retr_texts, retr_path, retriever)
   
def _get_amazon_retr_data(data):
    queries = []
    retr_text = []
    retr_gts = []
    for sample in data:
        queries.append(sample["Review"])
        retr_text.append([item["Review"] for item in sample["History"]])
        retr_gts.append([item["Score"] for item in sample["History"]])
    return queries, retr_text, retr_gts

def _get_lamp_retr_data(data, dataset_num=5):
    queries = []
    retr_text = []
    retr_gts = []
    prof_text_name, prof_gt_name, _  = get_lamp_var_names(dataset_num)
    for sample in data:
        if dataset_num in [3, 4, 5, 7]:
            text_idx = sample["input"].find(":") + 1
            queries.append(sample["input"][text_idx:].strip())
        elif dataset_num  == 1:
            queries.append(sample["input"].strip())
        elif dataset_num == 2:
            text_idx = sample["input"].find("description:") + 1
            queries.append(sample["input"][text_idx+len("description:"):].strip())
        retr_text.append([p[prof_text_name] for p in sample["profile"]])
        if dataset_num != 7:
            retr_gts.append([p[prof_gt_name] for p in sample["profile"]])
        else:
            retr_gts = retr_text
    return queries, retr_text, retr_gts

def _find_retr_res(queries, retr_texts, out_path, model="bm25", device="cuda:0"):

    retr_doc_idxs = []
    if model == "bm25":
        for i in range(len(retr_texts)):
            bm25 = BM25Okapi(retr_texts[i])
            doc_scores = bm25.get_scores(queries[i])
            retr_doc_idxs.append(doc_scores.argsort()[::-1])
    elif model in ["contriever", "dpr"]:
        if model == "contriever":
            retr_model = Contriever.from_pretrained("facebook/contriever-msmarco") 
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        elif model == "dpr":
            retr_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")  
        retr_model.to(device).eval()
        with torch.no_grad():
            for i in range(len(retr_texts)):
                inp = retr_texts[i]
                inp.append(queries[i]) 
                inputs = tokenizer(inp, padding=True, truncation=True, return_tensors="pt")
                inputs.to(device)
                embeddings = retr_model(**inputs)
                if model == "dpr":
                    embeddings = embeddings.pooler_output
                embeddings = embeddings.cpu()
                sim_scores = np.dot(embeddings[-1:], embeddings[:-1].T)    
                sorted_idxs = np.argsort(sim_scores).squeeze()[::-1]
                retr_doc_idxs.append(sorted_idxs)
    else:
        raise Exception("Retriever not implemented!")     
    with open(out_path, "w") as f:
        json.dump(retr_doc_idxs, f)
    return retr_doc_idxs

def _prepare_lamp_context(retr_texts, retr_gts, retr_doc_idxs, k, dataset_num):
    skip_k = 0
    doc_k = k
    _, prof_gt_name, prof_prompt_name = get_lamp_var_names(dataset_num)
    if "_" in k:
        doc_k = k.split("_")[0]
        if "skip" in k:
            skip_k = int(k[k.find("skip_")+len("skip_")])
    all_examples = []
    for i in range(len(retr_texts)):
        retr_docs = retr_doc_idxs[i]
        if "max" in k:
            doc_k = len(retr_docs)-skip_k
        else:
            doc_k = int(doc_k)
        retr_texts = [retr_texts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
        retr_gts = [retr_gts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
        if k.endswith("shuffle"):
            retr_texts, retr_gts = shuffle_lists(retr_texts, retr_gts)
        if k.endswith("reverse"):
            retr_texts = retr_texts[::-1]
            retr_gts = retr_gts[::-1]
        examples = []
        for text, gt in zip(retr_texts, retr_gts):
            if dataset_num != 7:
                example = f"{prof_prompt_name.capitalize()}:\n{text}\n{prof_gt_name.capitalize()}:\n{gt}\n" 
            else:
                example = f"{prof_prompt_name.capitalize()}:\n{text}"
            examples.append(example)
        all_examples.append(examples)
    return all_examples