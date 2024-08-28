import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, model: str = "bm25", device: str = "cuda:0"):
        self.model = model
        self.device = device
        self._init_model()

    def _init_model(self):
        if self.model != "bm25":
            if self.model == "contriever":
                self.model = "nishimoto/contriever-sentencetransformer"
            elif self.model == "dpr":
                self.model = "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base"

            self.retr_model = SentenceTransformer(self.model)

    def get_retrieval_results(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        retr_doc_idxs = []
        
        if self.model == "bm25":
            retr_doc_idxs = self._bm25_retrieval(queries, retr_texts)
        else:
            retr_doc_idxs = self._neural_retrieval(queries, retr_texts)

        return retr_doc_idxs

    def _bm25_retrieval(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:
        retr_doc_idxs = []
        for i, query in enumerate(queries):
            bm25 = BM25Okapi(retr_texts[i])
            doc_scores = bm25.get_scores(query)
            retr_doc_idxs.append(doc_scores.argsort()[::-1].tolist())
        return retr_doc_idxs

    def _encode(self, docs):
        if not isinstance(docs, np.ndarray):
            return self.retr_model.encode(docs)
        else:
            return docs

    def _neural_retrieval(self, queries: List[str], docs: List[str]):

        query_embeds = self._encode(queries)
        doc_embeds = self._encode(docs)
        similarities = self.retr_model.similarity(query_embeds, doc_embeds).numpy().squeeze()
        sorted_idxs = np.argsort(similarities)[::-1] 

        return sorted_idxs