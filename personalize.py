from typing import List, Union
import re
import os
import json
from abc import ABC, abstractmethod

import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

from utils import shuffle_lists

class Personalize(ABC):
    def __init__(self, dataset, retriever) -> None:
        self.dataset = dataset
        self.retriever = retriever
        
    @abstractmethod
    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        pass

    def parse_k(self, k):
        skip_k = 0
        doc_k = k
        if "_" in k:
            doc_k = k.split("_")[0]
            if "skip" in k:
                skip_k = int(k[k.find("skip_")+len("skip_"):])  
        return int(doc_k), skip_k

class RAG(Personalize):
    def __init__(self, dataset, retriever) -> None:
        super().__init__(dataset, retriever)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        doc_k, skip_k = self.parse_k(k)
        retr_path = "retrieval_res"
        os.makedirs(retr_path, exist_ok=True)
        file_path = os.path.join(retr_path, f"{self.dataset.tag}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_retr_docs = json.load(f)
                save = False
        else:
            save = True
            all_retr_docs = []
        all_examples = []
        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()
        for i, query in enumerate(queries):
            retr_text = retr_texts[i]
            retr_gt = retr_gts[i]
            if not save:
                retr_docs = np.array(all_retr_docs[i])
            else:
                retr_docs = self.retriever.get_retrieval_results(query, retr_text)
                all_retr_docs.append(retr_docs)
            
            texts = [retr_text[doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            gts = [retr_gt[doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            
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

        if save:
            with open(file_path, "w") as f:
                all_retr_docs = [d.tolist() for d in all_retr_docs]
                json.dump(all_retr_docs, f)
        return all_examples

    def prepare_prompt(self, method, query, llm, examples=None):
        init_prompt = self.dataset.get_prompt(method)
        zero_shot = init_prompt.format(query=query, examples="")
        if not examples:
            return zero_shot
        else:
            context = llm.prepare_context(zero_shot, examples)    
            return init_prompt.format(query=query, examples=context)
    
class CW_Map(Personalize):
    def __init__(self, dataset, retriever) -> None:
        super().__init__(dataset, retriever)
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
                            embedding = self.retriever._encode(context_sentences)
                            
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
    
    def get_word_distances(self, query, word_corpus):
        word_embeds = []
        for word in word_corpus.keys():
            word_embeds.append(np.array([s["embedding"] for s in word_corpus[word]["sentences"]]).mean(axis=0, dtype=np.float32))
        word_embeds = np.array(word_embeds)
        return self.retriever.get_retrieval_results(query, word_embeds)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:

        words = []
        for i, query in enumerate(queries):
            profile = self.get_profile(retr_texts[i], retr_gts[i])
            word_corpus = self.process_corpus(profile)
            sorted_idxs = self.get_word_distances(query, word_corpus)
            sorted_words = [list(word_corpus)[idx] for idx in sorted_idxs[:int(k)]]
            words.append(sorted_words)
        return words

    def prepare_prompt(self, method, query, examples, llm):
        init_prompt = self.dataset.get_prompt(method)
        context = llm.prepare_context(init_prompt, examples) 
        return init_prompt.format(query=query, words=context)

def get_personalization_method(method: str, dataset, retriever) -> Personalize:
    if method == "RAG":
        return RAG(dataset, retriever)
    elif method == "CW_Map":
        return CW_Map(dataset, retriever)
    else:
        raise ValueError(f"Unknown personalization method: {method}")