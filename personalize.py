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

from utils import shuffle_lists, softmax

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
            retr_text = retr_texts[i]
            retr_gt = retr_gts[i]
            if len(all_idxs) > i:
                similarities = all_similarities[i]
                sorted_idxs = np.array(all_idxs[i])
            else:
                similarities, sorted_idxs = self.retriever.get_retrieval_results(query, retr_text)
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

    def prepare_prompt(self, method, query, llm, examples=None):
        init_prompt = self.dataset.get_prompt(method)
        zero_shot = init_prompt.format(query=query, examples="")
        if not examples:
            return zero_shot
        else:
            context = llm.prepare_context(zero_shot, examples)    
            return init_prompt.format(query=query, examples=context)
    
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

    def process_profile(self, retr_texts, retr_gts, include_embeds=False, context_size=1):
        profile = [f"{gt}: {text}" for text, gt in zip(retr_texts, retr_gts)]
        word_info = {}
        stop_words = set(stopwords.words('english'))
        embedding_cache = {}
        contexts_to_embed = set()

        for doc_id, document in enumerate(profile):
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
                            word_info[lemma]["sentences"][context_sentences] = {
                                "doc_id": doc_id,
                                "position": [f"{start}:{end}"],
                            }
                            if include_embeds:
                                contexts_to_embed.add(context_sentences)
                        else:
                            word_info[lemma]["sentences"][context_sentences]["position"].append(f"{start}:{end}")

        if include_embeds and contexts_to_embed:
            batched_embeddings = self.retriever._encode(list(contexts_to_embed))
            embedding_cache = dict(zip(contexts_to_embed, batched_embeddings))

        for lemma in word_info:
            word_info[lemma]["original_forms"] = list(word_info[lemma]["original_forms"])
            word_info[lemma]["sentences"] = [
                {
                    "sentence": sentence,
                    "doc_id": info["doc_id"],
                    "position": info["position"],
                    **({'embedding': embedding_cache[sentence].tolist()} if include_embeds else {})
                }
                for sentence, info in word_info[lemma]["sentences"].items()
            ]

        return word_info
    
    def get_word_distances(self, query, retr_texts, retr_gts):
        profile_words = self.process_profile(retr_texts, retr_gts, include_embeds=True)
        word_embeds = []
        for word in profile_words.keys():
            word_embeds.append(np.array([s["embedding"] for s in profile_words[word]["sentences"]]).mean(axis=0, dtype=np.float32))
        word_embeds = np.array(word_embeds)
        return self.retriever.get_retrieval_results(query, word_embeds)

    def get_profile_words(self, retr_texts, retr_gts):
        return self.process_profile(retr_texts, retr_gts)

    def get_classes(self, retr_gts):
            new_classes = []
            classes = list(set(retr_gts))
            for cls in classes:
                cls_idxs = [i for i, gt in enumerate(retr_gts) if gt == cls]
                if len(cls_idxs) >= 2:
                    new_classes.append(cls)
            return new_classes
    
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
            return self.get_word_distances(query, retr_texts, retr_gts)
        
    def get_sorted_words(self, retr_texts, retr_gts, sorted_idxs, similarities=None):
        if self.dataset.task == "classification" and self.dataset.num != 1:
            classes = self.get_classes(retr_gts)
            return [f"{classes[idx]}, {similarities[idx]})" for idx in sorted_idxs]
        else:
            profile_words = self.get_profile_words(retr_texts, retr_gts)
            return [list(profile_words)[idx] for idx in sorted_idxs]  

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
            sorted_words = self.get_sorted_words(retr_texts[i], retr_gts[i], sorted_idxs, similarities)
            words.append(sorted_words[:int(k)])         
        return words

    def prepare_prompt(self, method, query, llm, examples):
        init_prompt = self.dataset.get_prompt(method)
        context = llm.prepare_context(init_prompt, examples) 
        return init_prompt.format(query=query, words=context.splitlines())


class Comb(Personalize):
    def __init__(self, dataset, retriever) -> None:
        super().__init__(dataset, retriever)


def get_personalization_method(method: str, dataset, retriever) -> Personalize:
    if method == "RAG":
        return RAG(dataset, retriever)
    elif method == "CWMap":
        return CWMap(dataset, retriever)
    elif method == "Comb":
        return Comb(dataset, retriever)
    else:
        raise ValueError(f"Unknown personalization method: {method}")
    
