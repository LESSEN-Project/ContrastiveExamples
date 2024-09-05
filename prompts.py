def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())

def amazon_prompts(method):
    if method == "RAG":
        return _RAG_amazon_prompt()
    else:
        raise Exception("No such method exists!")

def _RAG_amazon_prompt() -> str:
    return strip_all("""Your task is to generate a review for a product the customer bought.
                     You will be provided some similar product name-review pairs from the customer's past purchases to help you with the task.
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Capture the customer's writing style while generating the review. Only output the review and nothing else.
                     Product Name:
                     {query}
                     Review:""")

def lamp_prompts(dataset_num: int, method: str) -> str:
    if method == "RAG":
        RAG_lamp_prompts = {
            4: _RAG_lamp_prompt_4,
            5: _RAG_lamp_prompt_5,
            7: _RAG_lamp_prompt_7
        }
        return RAG_lamp_prompts.get(dataset_num)()

def _RAG_lamp_prompt_4() -> str:
    return strip_all("""Your task is to generate a title for the given news article.
                     You will be provided some similar article-title pairs from editor's past works to help you with the task.
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Capture the news editor's writing style while generating the title. Only output the title and nothing else.
                     Article: 
                     {query}
                     Title:""")

def _RAG_lamp_prompt_5() -> str:
    return strip_all("""Your task is to generate a title for the given academic abstract.
                     You will be provided some similar abstract-title pairs from scholar's past works to help you with the task.
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Capture the scholar's writing style while generating the title. Only output the title and nothing else.
                     Abstract:
                     {query}
                     Title:""")

def _RAG_lamp_prompt_7() -> str:
    return strip_all("""Your task is to rephrase a tweet in the style of the user.
                     You will be provided some similar tweets from user's past tweets to help you with the task.
                     <SimilarTweets>
                     {examples}
                     </SimilarTweets>
                     Capture user's writing style while paraphrasing. Only output the tweet and nothing else.
                     Tweet:
                     {query}
                     Paraphrased Tweet:""")



"""Your task is to generate a title for the given news article.
                     You will be provided some similar article-title pairs from editor's past works to help you with the task.
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Capture the news editor's writing style while generating the title. Only output the title and nothing else.
                     Article: 
                     {query}
                     Title:"""