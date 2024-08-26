from typing import List, Dict

def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())

def amazon_prompt() -> str:
    return strip_all("""Here are a couple of product reviews of an amazon user:
                  <EXAMPLES>
                  {examples}
                  </EXAMPLES>
                  With the given examples, generate review for the given product purchased by the same user. Only output the review and nothing else.
                  Product Name:
                  {query}
                  Review:
                  """)

def lamp_prompts(dataset_num: int) -> List[Dict[str, str]]:
    lamp_prompts = {
        1: _lamp_prompt_1,
        2: _lamp_prompt_2,
        3: _lamp_prompt_3,
        4: _lamp_prompt_4,
        5: _lamp_prompt_5,
        7: _lamp_prompt_7
    }
    return lamp_prompts.get(dataset_num)()

def _lamp_prompt_1() -> str:
    return strip_all("""Here are a couple of abstract-title pairs of a scholar.
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, complete the following task. Only output the response of the task and nothing else.
                    Task:
                    {query}
                    """)

def _lamp_prompt_2() -> str:    
    return strip_all("""Here are a couple of movie description-tag pairs.
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, choose the correct category tag for the following movie description between these tags:
                    [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                    Only output the tag and nothing else.
                    Description:
                    {query}
                    Tag:""")

def _lamp_prompt_3() -> str:
    return strip_all("""Here are a couple of review-rating pairs of a user. 
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, give a score between [1, 2, 3, 4, 5] to the following review by the same user. Only output the score and nothing else.
                    Review:
                    {query}
                    Score:""")

def _lamp_prompt_4() -> str:
    return strip_all("""Here are a couple of article-title pairs of a user. 
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, generate a title for the given article by the same author. Only output the title and nothing else.
                    Article: 
                    {query}
                    Title:""")

def _lamp_prompt_5() -> str:
    return strip_all("""Here are a couple of abstract-title pairs of a scholar:
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, generate a title for the given abstract by the same author. Only output the title and nothing else.
                    Abstract:
                    {query}
                    Title:""")

def _lamp_prompt_7() -> str:
    return strip_all("""Here are a couple of tweets of a person:
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, paraphrase the given tweet by the same person. Only output the tweet and nothing else.
                    Tweet:
                    {query}
                    Paraphrased Tweet:""")