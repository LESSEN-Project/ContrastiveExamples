from typing import List
from utils import shuffle_lists

def get_context(retr_texts: List[List[str]], retr_gts: List[List[str]], retr_doc_idxs: List[List[int]], 
                k: str, retr_text_name: str, retr_gt_name: str) -> List[List[str]]:
    skip_k = 0
    doc_k = k
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
                example = f"{retr_text_name.capitalize()}:\n{text}\n{retr_gt_name.capitalize()}:\n{gt}\n"
            else:
                example = f"{retr_text_name.capitalize()}:\n{text}"
            examples.append(example)
        all_examples.append(examples)
    
    return all_examples