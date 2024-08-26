from typing import List, Union
import re
from abc import ABC, abstractmethod

import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer

from utils import shuffle_lists

class Personalize(ABC):
    @abstractmethod
    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        pass

class RAG(Personalize):
    def __init__(self, dataset, retriever) -> None:
        self.dataset = dataset
        self.retriever = retriever

    def prepare_retrieval_results(self, queries, retr_texts):
        return self.retriever.get_retrieval_results(self.dataset.tag, queries, retr_texts)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        if k == "0":
            return None
        retr_doc_idxs = self.prepare_retrieval_results(queries, retr_texts)
        skip_k = 0
        doc_k = k
        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()
        if "_" in k:
            doc_k = k.split("_")[0]
            if "skip" in k:
                skip_k = int(k[k.find("skip_")+len("skip_"):])
        
        all_examples = []
        for i, retr_docs in enumerate(retr_doc_idxs):
            if "max" in k:
                doc_k = len(retr_docs) - skip_k
            else:
                doc_k = int(doc_k)
            
            texts = [retr_texts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            gts = [retr_gts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            
            if k.endswith("shuffle"):
                texts, gts = shuffle_lists(texts, gts)
            if k.endswith("reverse"):
                texts = texts[::-1]
                gts = gts[::-1]
            
            examples = []
            for text, gt in zip(texts, gts):
                if text != gt:
                    example = f"{retr_prompt_name.capitalize()}:\n{text}\n{retr_gt_name.capitalize()}:\n{gt}\n"
                else:
                    example = f"{retr_prompt_name.capitalize()}:\n{text}"
                examples.append(example)
            all_examples.append(examples)
        
        return all_examples
    
class CW_Map(Personalize):
    def __init__(self, dataset, retriever, model="all-MiniLM-L6-v2") -> None:
        self.dataset = dataset
        self.retriever = retriever
        self.model = SentenceTransformer(model)
        self.download_nltk()

    def download_nltk(self):
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            words = [match.group() for match in re.finditer(r'\b[a-zA-Z]+\b', sentence)]
            pos_tags = pos_tag(words)
            
            words_with_positions = []
            current_position = 0
            
            for word, pos in pos_tags:
                start = sentence.lower().find(word.lower(), current_position)
                end = start + len(word)
                current_position = end
                
                lemma = lemmatizer.lemmatize(word.lower(), pos=self.get_wordnet_pos(pos))
                words_with_positions.append((word.lower(), lemma, start, end))
            
            processed_sentences.append((sentence, words_with_positions))
        
        return processed_sentences, sentences

    def process_corpus(self, corpus, context_size=1):
        word_info = {}
        stop_words = set(stopwords.words('english'))
        
        for doc_id, document in enumerate(corpus):
            processed_sentences, original_sentences = self.preprocess_text(document)
            
            for sent_idx, (_, words_with_positions) in enumerate(processed_sentences):
                for original, lemma, start, end in words_with_positions:
                    if lemma not in stop_words:
                        if lemma not in word_info:
                            word_info[lemma] = {
                                "count": 0,
                                "sentences": {},
                                "original_forms": set()
                            }
                        
                        word_info[lemma]["count"] += 1
                        word_info[lemma]["original_forms"].add(original)
                        
                        start_idx = max(0, sent_idx - context_size)
                        end_idx = min(len(original_sentences), sent_idx + context_size + 1)
                        context_sentences = " ".join(original_sentences[start_idx:end_idx])
                        
                        if context_sentences not in word_info[lemma]["sentences"]:
                            embedding = self.model.encode(context_sentences)
                            
                            word_info[lemma]["sentences"][context_sentences] = {
                                "doc_id": doc_id,
                                "position": [f"{start}:{end}"],
                                "embedding": embedding.tolist() 
                            }
                        else:
                            word_info[lemma]["sentences"][context_sentences]["position"].append(f"{start}:{end}")

        for lemma in word_info:
            word_info[lemma]["original_forms"] = list(word_info[lemma]["original_forms"])
            word_info[lemma]["sentences"] = [
                {
                    "sentence": sentence,
                    "doc_id": info["doc_id"],
                    "position": info["position"],
                    "embedding": info["embedding"]
                }
                for sentence, info in word_info[lemma]["sentences"].items()
            ]

        return word_info

    def get_profile(self, retr_texts, retr_gts):
        return [f"{gt}: {text}" for text, gt in zip(retr_texts, retr_gts)]
    
    def get_word_distances(self, queries, word_corpus):
        word_embeds = []
        for word in word_corpus.keys():
            word_embeds.append(np.array([s["embedding"] for s in word_corpus[word]["sentences"]]).mean(axis=0, dtype=np.float32))
        word_embeds = np.array(word_embeds)
        query_embeds = self.model.encode(queries)
        similarities = self.model.similarity(query_embeds, np.array(word_embeds)).numpy().squeeze()
        sorted_idxs = np.argsort(similarities)[::-1]
        return similarities, sorted_idxs

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        profile = self.get_profile(retr_texts, retr_gts)
        word_corpus = word_corpus(profile)
        similarities, sorted_idxs = self.get_word_distances(queries, word_corpus)
        sorted_words = [word[:k] for word in sorted_idxs]
        words = []
        for word in sorted_words:
            words.append([list(word_corpus.values())[idx]["original_forms"] for idx in word])


def get_personalization_method(method: str, dataset, retriever, model="all-MiniLM-L6-v2") -> Personalize:
    if method == "RAG":
        return RAG(dataset, retriever)
    elif method == "CW_Map":
        return CW_Map(dataset, retriever, model)
    else:
        raise ValueError(f"Unknown personalization method: {method}")