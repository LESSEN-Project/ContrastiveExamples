from typing import List, Union
import os
import json
from abc import ABC, abstractmethod


class Personalizer(ABC):
    def __init__(self, dataset, retriever) -> None:
        self.dataset = dataset
        self.retriever = retriever
        self.save_loc = "retrieval_res"
        os.makedirs(self.save_loc, exist_ok=True)

    def check_file(self):
        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{self.retriever.model}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_idxs = json.load(f)
        else:
            print("Retrieval results are not cached, starting from 0!")
            all_idxs = []
        return all_idxs
    
    def save_file(self, obj):
        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{self.retriever.model}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> List[List[str]]:
        all_idxs = self.check_file()
        all_examples = []
        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()
        for i, query in enumerate(queries):
            if isinstance(query, list):
                query = query[0]
            retr_text = retr_texts[i]
            retr_gt = retr_gts[i]
            if len(all_idxs) > i:
                sorted_idxs = all_idxs[i]
            else:
                if self.dataset.num == 1:
                    retr_var = retr_gt
                else:
                    retr_var = retr_text
                _, sorted_idxs = self.retriever.get_retrieval_results(query, retr_var)
                all_idxs.append(sorted_idxs)
                if ((i+1)%500 == 0) or (i+1 == len(queries)):
                    print(i)     
                    self.save_file(all_idxs)
            
            texts = [retr_text[doc_id] for doc_id in sorted_idxs[:k]]
            gts = [retr_gt[doc_id] for doc_id in sorted_idxs[:k]]
            
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