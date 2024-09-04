from typing import List, Union
import re
import os
import json
from abc import ABC, abstractmethod

import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

from utils import shuffle_lists

class Personalize(ABC):
    def __init__(self, dataset, retriever) -> None:
        self.dataset = dataset
        self.retriever = retriever
        self.save_loc = "retrieval_res"
        os.makedirs(self.save_loc, exist_ok=True)
        
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

    def check_file(self, method):
        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{method}_{self.retriever.model}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_res = json.load(f)
                all_similaries = all_res[0]
                all_idxs = all_res[1]
        else:
            print("Retrieval results are not cached, starting from 0!")
            all_similaries = []
            all_idxs = []
        return all_similaries, all_idxs
    
    def save_file(self, method, obj):
        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{method}_{self.retriever.model}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

class RAG(Personalize):
    def __init__(self, dataset, retriever) -> None:
        super().__init__(dataset, retriever)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:
        doc_k, skip_k = self.parse_k(k)
        all_similarities, all_idxs = self.check_file("RAG")
        all_examples = []
        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()
        for i, query in enumerate(queries):
            if isinstance(query, list):
                query = query[0]
            retr_text = retr_texts[i]
            retr_gt = retr_gts[i]
            if len(all_idxs) > i:
                similarities = all_similarities[i]
                sorted_idxs = np.array(all_idxs[i])
            else:
                if self.dataset.num == 1:
                    retr_var = retr_gt
                else:
                    retr_var = retr_text
                similarities, sorted_idxs = self.retriever.get_retrieval_results(query, retr_var)
                all_similarities.append(similarities)
                all_idxs.append(sorted_idxs)
                if ((i+1)%500 == 0) or (i+1 == len(queries)):
                    print(i)     
                    self.save_file("RAG", (all_similarities, all_idxs))
            
            texts = [retr_text[doc_id] for doc_id in sorted_idxs[skip_k: (doc_k+skip_k)]]
            gts = [retr_gt[doc_id] for doc_id in sorted_idxs[skip_k: (doc_k+skip_k)]]
            
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

    def prepare_prompt(self, method, query, llm, examples):
        init_prompt = self.dataset.get_prompt(method)
        if self.dataset.num != 1:
            context = llm.prepare_context(init_prompt, query, examples)
            return init_prompt.format(query=query, examples=context)
        else:
            context = llm.prepare_context(init_prompt, str(query), examples)
            real_query = query[0]
            first_option = query[1]
            second_option = query[2]
            return init_prompt.format(query=real_query, examples=context, first_option=first_option, second_option=second_option)
    
class CWMap(Personalize):
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
        words = word_tokenize(text.lower())
        processed_words = []
        current_position = 0
        
        for word in words:
            if word.isalnum():
                start = text.lower().find(word, current_position)
                end = start + len(word)
                current_position = end
                
                lemma = lemmatizer.lemmatize(word)
                processed_words.append((word, lemma, start, end))
        
        return processed_words

    def process_profile(self, retr_texts, retr_gts, include_embeds=False, context_size=3):
        profile = [f"{gt}: {text}" for text, gt in zip(retr_texts, retr_gts)]
        word_info = {}
        stop_words = set(stopwords.words('english'))
        embedding_cache = {}
        contexts_to_embed = set()

        for doc_id, document in enumerate(profile):
            processed_words = self.preprocess_text(document)
            
            for word_idx, (original, lemma, start, end) in enumerate(processed_words):
                if lemma not in stop_words:
                    if lemma not in word_info:
                        word_info[lemma] = {
                            "count": 0,
                            "contexts": {},
                            "original_forms": set()
                        }
                    
                    word_info[lemma]["count"] += 1
                    word_info[lemma]["original_forms"].add(original)
                    
                    start_idx = max(0, word_idx - context_size)
                    end_idx = min(len(processed_words), word_idx + context_size + 1)
                    context_words = [w[0] for w in processed_words[start_idx:end_idx]]
                    context = " ".join(context_words)
                    
                    if context not in word_info[lemma]["contexts"]:
                        word_info[lemma]["contexts"][context] = {
                            "doc_id": doc_id,
                            "position": [f"{start}:{end}"],
                        }
                        if include_embeds:
                            contexts_to_embed.add(context)
                    else:
                        word_info[lemma]["contexts"][context]["position"].append(f"{start}:{end}")

        if include_embeds and contexts_to_embed:
            batched_embeddings = self.retriever._encode(list(contexts_to_embed))
            embedding_cache = dict(zip(contexts_to_embed, batched_embeddings))

        for lemma in word_info:
            word_info[lemma]["original_forms"] = list(word_info[lemma]["original_forms"])
            word_info[lemma]["contexts"] = [
                {
                    "context": context,
                    "doc_id": info["doc_id"],
                    "position": info["position"],
                    **({'embedding': embedding_cache[context].tolist()} if include_embeds else {})
                }
                for context, info in word_info[lemma]["contexts"].items()
            ]

        return word_info
    
    def get_word_distances(self, query, retr_texts, retr_gts):
        profile_words = self.process_profile(retr_texts, retr_gts, include_embeds=True)
        word_embeds = []
        for word in profile_words.keys():
            word_embeds.append(np.array([s["embedding"] for s in profile_words[word]["contexts"]]).mean(axis=0, dtype=np.float32))
        word_embeds = np.array(word_embeds)
        return self.retriever.get_retrieval_results(query, word_embeds)

    def get_profile_words(self, retr_texts, retr_gts):
        return self.process_profile(retr_texts, retr_gts)

    def get_classes(self, retr_gts):
        return list(set(retr_gts))
    
    def get_class_distances(self, query, retr_texts, retr_gts):
        classes = self.get_classes(retr_gts)
        all_cls_texts = []
        for cls in classes:
            cls_idxs = [i for i, gt in enumerate(retr_gts) if gt == cls]
            cls_texts = self.retriever._encode([retr_texts[idx] for idx in cls_idxs])
            all_cls_texts.append(np.array(cls_texts.mean(axis=0, dtype=np.float32)))
        return self.retriever.get_retrieval_results(query, np.array(all_cls_texts))

    def get_distances(self, query, retr_texts, retr_gts):
        if self.dataset.task == "classification" and self.dataset.num != 1:
            return self.get_class_distances(query, retr_texts, retr_gts)
        else:
            if self.dataset.num == 1:
                query = query[0]
            return self.get_word_distances(query, retr_texts, retr_gts)
        
    def get_sorted_words(self, retr_texts, retr_gts, sorted_idxs, similarities, k):
        if self.dataset.task == "classification" and self.dataset.num != 1:
            classes = self.get_classes(retr_gts)
            return [f"{classes[idx]}, {round(similarities[idx], 3)}" for idx in sorted_idxs]
        else:
            profile_words = self.get_profile_words(retr_texts, retr_gts)
            # return [list(profile_words)[idx] for idx in sorted_idxs][:int(k)]
            return [f"{list(profile_words)[idx]}, {round(similarities[idx], 3)}" for idx in sorted_idxs][:int(k)]

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> Union[List[List[str]], None]:

        all_similarities, all_idxs = self.check_file("CWMap")
        words = []
        for i, query in enumerate(queries):
            if len(all_idxs) > i:
                similarities = all_similarities[i]
                sorted_idxs = all_idxs[i]
            else:
                similarities, sorted_idxs = self.get_distances(query, retr_texts[i], retr_gts[i])
                all_idxs.append(sorted_idxs)
                all_similarities.append(similarities)
                if ((i+1)%500 == 0) or (i+1 == len(queries)):
                    print(i)     
                    self.save_file("CWMap", (all_similarities, all_idxs))
            sorted_words = self.get_sorted_words(retr_texts[i], retr_gts[i], sorted_idxs, similarities, k)
            words.append(sorted_words)
        return words

    def prepare_prompt(self, method, query, llm, examples):
        init_prompt = self.dataset.get_prompt(method)
        if self.dataset.num != 1:
            context = llm.prepare_context(init_prompt, query, examples) 
            words = context.splitlines()
            return init_prompt.format(query=query, words=words)
        else:
            context = llm.prepare_context(init_prompt, str(query), examples) 
            words = context.splitlines()
            real_query = query[0]
            first_option = query[1]
            second_option = query[2]     
            return init_prompt.format(query=real_query, words=words, first_option=first_option, second_option=second_option)       


class Comb(Personalize):
    def __init__(self, dataset, retriever) -> None:
        super().__init__(dataset, retriever)
        self.rag_module = RAG(dataset, retriever)
        self.cw_module = CWMap(dataset, retriever)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> List[List[str]] | None:
        rag_context = self.rag_module.get_context(queries, retr_texts, retr_gts, k[0])
        cw_context = self.cw_module.get_context(queries, retr_texts, retr_gts, k[1])
        return [context for context in zip(rag_context, cw_context)]
    
    def prepare_prompt(self, method, query, llm, examples):
        init_prompt = self.dataset.get_prompt(method)
        if self.dataset.num != 1:
            cw_context = llm.prepare_context(init_prompt, query, examples[1])
            rag_context = llm.prepare_context(init_prompt, f"{query}c\n{examples[1]}", examples[0])
            return init_prompt.format(query=query, words=cw_context.splitlines(), examples=rag_context)
        else:
            cw_context = llm.prepare_context(init_prompt, str(query), examples[1])
            rag_context = llm.prepare_context(init_prompt, f"{query}c\n{examples[1]}", examples[0])      
            real_query = query[0]
            first_option = query[1]
            second_option = query[2]         
            return init_prompt.format(query=real_query, words=cw_context.splitlines(), examples=rag_context, first_option=first_option, second_option=second_option)

def get_personalization_method(method: str, dataset, retriever) -> Personalize:
    if method == "RAG":
        return RAG(dataset, retriever)
    elif method == "CWMap":
        return CWMap(dataset, retriever)
    elif method == "Comb":
        return Comb(dataset, retriever)
    else:
        raise ValueError(f"Unknown personalization method: {method}")
    
