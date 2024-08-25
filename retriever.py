import os
import json
import torch
import numpy as np
from typing import List

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from contriever.src.contriever import Contriever

class Retriever:
    def __init__(self, model: str = "bm25", device: str = "cuda:0"):
        self.model = model
        self.device = device
        self._init_model()

    def _init_model(self):
        if self.model == "contriever":
            self.retr_model = Contriever.from_pretrained("facebook/contriever-msmarco")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        elif self.model == "dpr":
            self.retr_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        if self.model in ["contriever", "dpr"]:
            self.retr_model.to(self.device).eval()

    def get_retrieval_results(self, dataset_info: str, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        retr_path = "retrieval_res"
        os.makedirs(retr_path, exist_ok=True)
        file_path = os.path.join(retr_path, f"{dataset_info}.json")
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return self._find_retrieval_results(queries, retr_texts, file_path)

    def _find_retrieval_results(self, queries: List[str], retr_texts: List[List[str]], out_path: str) -> List[List[int]]:
        retr_doc_idxs = []
        
        if self.model == "bm25":
            retr_doc_idxs = self._bm25_retrieval(queries, retr_texts)
        elif self.model in ["contriever", "dpr"]:
            retr_doc_idxs = self._neural_retrieval(queries, retr_texts)
        else:
            raise ValueError("Retriever not implemented!")
        
        with open(out_path, "w") as f:
            json.dump(retr_doc_idxs, f)
        
        return retr_doc_idxs

    def _bm25_retrieval(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        retr_doc_idxs = []
        for i, query in enumerate(queries):
            bm25 = BM25Okapi(retr_texts[i])
            doc_scores = bm25.get_scores(query)
            retr_doc_idxs.append(doc_scores.argsort()[::-1].tolist())
        return retr_doc_idxs

    def _neural_retrieval(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        retr_doc_idxs = []
        with torch.no_grad():
            for i, query in enumerate(queries):
                inp = retr_texts[i] + [query]
                inputs = self.tokenizer(inp, padding=True, truncation=True, return_tensors="pt")
                inputs = inputs.to(self.device)
                embeddings = self.retr_model(**inputs)
                if self.model == "dpr":
                    embeddings = embeddings.pooler_output
                embeddings = embeddings.cpu()
                sim_scores = np.dot(embeddings[-1:], embeddings[:-1].T)
                sorted_idxs = np.argsort(sim_scores)
                if len(sorted_idxs) > 1:
                    sorted_idxs = sorted_idxs.squeeze()[::-1]
                retr_doc_idxs.append(sorted_idxs.tolist())
        return retr_doc_idxs