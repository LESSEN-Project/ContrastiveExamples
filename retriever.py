import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model: str = "contriever", device: str = "cuda:0"):
        self.model = model
        self.device = device
        self._init_model()

    def _init_model(self):
        if self.model == "contriever":
            self.retr_model = SentenceTransformer("nishimoto/contriever-sentencetransformer", device=self.device)
        elif self.model == "dpr":
            self.retr_model = SentenceTransformer("sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base", device=self.device)  
        else:
            self.retr_model = SentenceTransformer(self.model, device=self.device)

    def get_retrieval_results(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        return self._neural_retrieval(queries, retr_texts)

    def _encode(self, docs):
        if not isinstance(docs, np.ndarray):
            return self.retr_model.encode(docs)
        else:
            return docs

    def _neural_retrieval(self, queries: List[str], docs: List[str]):
        query_embeds = self._encode(queries)
        doc_embeds = self._encode(docs)
        similarities = self.retr_model.similarity(query_embeds, doc_embeds).numpy().squeeze().tolist()
        sorted_idxs = np.argsort(similarities)[::-1].tolist()
        if isinstance(similarities, float):
            similarities = [similarities]
        return similarities, sorted_idxs