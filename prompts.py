def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())

def amazon_prompts(method):
    if method == "RAG":
        return _RAG_amazon_prompt()
    elif method == "CWMap":
        return _CW_amazon_prompt()
    elif method == "Comb":
        return _Comb_amazon_prompt()
    else:
        raise Exception("No such method exists!")

def _RAG_amazon_prompt() -> str:
    return strip_all("""Here are a couple of product reviews of an Amazon customer:
                     <EXAMPLES>
                     {examples}
                     </EXAMPLES>
                     With the given examples, generate a review for the given product purchased by the customer. Only output the review and nothing else.
                     Product Name:
                     {query}
                     Review:""")

def _CW_amazon_prompt() -> str:
    return strip_all("""Your task is to generate a review for the product in customer's style.
                     Previous reviews of the customer are analyzed to understand which words the customer would use for this product.
                     Here is a list of potential words and their similarity scores with the given product:
                     {words}
                     The similarity scores are calculated by obtaning the word embeddings of the customer, then finding the distance between the words and the product.
                     Looking at the words, generate a review for the given product purchased by the customer. 
                     Remember that the rating-similarity scores are only estimations and may not always be representative, therefore you can use other words besides the ones listed.
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")

def _Comb_amazon_prompt():
    return strip_all("""Your task is to generate a review for the product in customer's style.
                     Previous reviews of the customer are analyzed to understand which words the customer would use for this product.
                     Here is a list of potential words and their similarity scores with the given product:
                     {words}
                     The similarity scores are calculated by obtaning the word embeddings of the customer, then finding the distance between the words and the product.
                     And here are a couple of product reviews of the customer:
                     <EXAMPLES>
                     {examples}
                     </EXAMPLES>
                     Looking at the used words and previous product reviews of the customer, generate a review for the following product purchased by the same customer.
                     Remember that the rating-similarity scores are only estimations and may not always be representative, therefore you can use other words besides the ones listed.
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")

def lamp_prompts(dataset_num: int, method: str) -> str:
    if method == "RAG":
        RAG_lamp_prompts = {
            1: _RAG_lamp_prompt_1,
            2: _RAG_lamp_prompt_2,
            3: _RAG_lamp_prompt_3,
            4: _RAG_lamp_prompt_4,
            5: _RAG_lamp_prompt_5,
            7: _RAG_lamp_prompt_7
        }
        return RAG_lamp_prompts.get(dataset_num)()
    elif method == "CWMap":
        CW_lamp_prompts = {
            1: _CW_lamp_prompt_1,
            2: _CW_lamp_prompt_2,
            3: _CW_lamp_prompt_3,
            4: _CW_lamp_prompt_4,
            5: _CW_lamp_prompt_5,
            7: _CW_lamp_prompt_7
        }
        return CW_lamp_prompts.get(dataset_num)()

def _CW_lamp_prompt_1() -> str:
    return strip_all("""Here is a list of words that a scholar uses frequently:
                    {words}
                    Looking at the words the scholar uses, complete the following task. You can use other words besides the ones listed, but give priority to them. Only output the response of the task and nothing else.
                    Task:
                    {query}
                    Response:""")

def _CW_lamp_prompt_2() -> str:
    return strip_all("""Your task is to predict the category tag of the movie.
                     Previous description-tag pairs of the user are analyzed to help you with the prediction. 
                     Here is the list of tags and the similarity scores they have with the given description:
                     {words}
                     The similarity scores are calculated by averaging the embeddings of descriptions for each category tag, then finding the distance between the averages and the given review.
                     Using the tag-similarity information, choose the correct category tag for the movie description between these tags:
                     [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                     Remember that the rating-similarity scores are only estimations and may not always be representative. Therefore, also use the movie description to improve your prediction accuracy.
                     You can choose a tag even though it is not present in user's previous tags. Only output the tag and nothing else.
                     Movie Description:
                     {query}
                     Tag:""")

def _CW_lamp_prompt_3() -> str:
    return strip_all("""Your task is to predict the rating user gave to the review.
                     Previous review-score pairs of the user are analyzed to help you with the prediction. 
                     Here is the list of ratings and the similarity scores they have with the given review:
                     {words}
                     The similarity scores are calculated by averaging the embeddings of reviews for each score category, then finding the distance between the averages and the given review.
                     Using the rating-similarity information, predict the score between [1, 2, 3, 4, 5] to the review. 
                     Remember that the rating-similarity scores are only estimations and may not always be representative. Therefore, also use the review to improve your prediction accuracy.
                     You can choose a tag even though it is not present in user's previous tags. Only output the score and nothing else.
                     Review:
                     {query}
                     Score:""")

def _CW_lamp_prompt_4() -> str:
    return strip_all("""Here is a list of words that an author frequently uses in their articles:
                    {words}
                    Looking at these words, generate a title for the given article. You can use other words besides the ones listed, but give priority to them. Only output the title and nothing else.
                    Article:
                    {query}
                    Title:""")

def _CW_lamp_prompt_5() -> str:
    return strip_all("""Here is a list of words that an author uses frequently:
                    {words}
                    Here is an abstract from the author:
                    {query}
                    Looking at the words the author uses, generate a title for the abstract. You can use other words besides the ones listed, but give priority to them. Only output the title and nothing else.
                    Title:""")

def _CW_lamp_prompt_7() -> str:
    return strip_all("""Here is a list of words that a person frequently uses in their tweets:
                    {words}
                    Looking at these words, paraphrase the given tweet. You can use other words besides the ones listed, but give priority to them. Only output the paraphrased tweet and nothing else.
                    Tweet:
                    {query}
                    Paraphrased Tweet:""")

def _RAG_lamp_prompt_1() -> str:
    return strip_all("""Here are a couple of abstract-title pairs of a scholar.
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, complete the following task. Only output the response of the task and nothing else.
                    Task:
                    {query}
                    """)

def _RAG_lamp_prompt_2() -> str:    
    return strip_all("""Here are a couple of movie description-tag pairs created by a user.
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, choose the correct category tag for the following movie description by the same user between these tags:
                    [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                    Only output the tag and nothing else.
                    Description:
                    {query}
                    Tag:""")

def _RAG_lamp_prompt_3() -> str:
    return strip_all("""Here are a couple of review-rating pairs of a user. 
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, give a score between [1, 2, 3, 4, 5] to the following review by the same user. Only output the score and nothing else.
                    Review:
                    {query}
                    Score:""")

def _RAG_lamp_prompt_4() -> str:
    return strip_all("""Here are a couple of article-title pairs of a user. 
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, generate a title for the given article by the same author. Only output the title and nothing else.
                    Article: 
                    {query}
                    Title:""")

def _RAG_lamp_prompt_5() -> str:
    return strip_all("""Here are a couple of abstract-title pairs of a scholar:
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, generate a title for the given abstract by the same author. Only output the title and nothing else.
                    Abstract:
                    {query}
                    Title:""")

def _RAG_lamp_prompt_7() -> str:
    return strip_all("""Here are a couple of tweets of a person:
                    <EXAMPLES>
                    {examples}
                    </EXAMPLES>
                    With the given examples, paraphrase the given tweet by the same person. Only output the tweet and nothing else.
                    Tweet:
                    {query}
                    Paraphrased Tweet:""")