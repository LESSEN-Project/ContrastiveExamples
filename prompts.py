def strip_all(text):
   return "\n".join([line.strip() for line in text.splitlines()])

def get_prompt(dataset_name, num, query, examples=None):
    if dataset_name == "lamp":
        return _lamp_prompt(num, query, examples)
    elif dataset_name == "amazon":
        return _amazon_prompt(query, examples)
    
def _amazon_prompt(query, examples):
    return [
        {"role": "user", "content": strip_all(f"""Here are a couple of product reviews of an amazon user:
                                                    <EXAMPLES>
                                                    {examples}
                                                    </EXAMPLES>
                                                    With the given examples, generate review for the given product purchased by the same user. Only output the review and nothing else.
                                                    Product Name:
                                                    {query}
                                                    Review:
                                                    """)}
    ]

def _lamp_prompt(dataset_num, query, examples):
    if dataset_num == 1:
            if examples:
                return [
                    {"role": "user", "content": strip_all(f"""Here are a couple of abstract-title pairs of a scholar.
                                                                <EXAMPLES>
                                                                {examples}
                                                                </EXAMPLES>
                                                                With the given examples, complete the following task. Only output the response of the task and nothing else.
                                                                Task:
                                                                {query}
                                                                """)}
                ]
            else:
                return [
                    {"role": "user", "content": strip_all(f"""Complete the following task. Only output the response of the task and nothing else.
                                                                Task:
                                                                {query}
                                                                """)}
                ]
            
    elif dataset_num == 2:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of movie description-tag pairs.
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, choose the correct category tag for the following movie description between these tags: 
                                                            [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                                                            Only output the tag and nothing else.
                                                            Description:
                                                            {query}
                                                            Tag:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Choose the correct category tag for the following movie description between these tags:
                                                            [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                                                            Only output the tag and nothing else.
                                                            Description:
                                                            {query}
                                                            Tag:""")}
            ]
        
    elif dataset_num == 3:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of review-rating pairs of a user. 
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, give a score between [1, 2, 3, 4, 5] to the following review by the same user. Only output the score and nothing else.
                                                            Review: 
                                                            {query}
                                                            Score:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Give a score between [1, 2, 3, 4, 5] to the following review. Only output the score and nothing else.
                                                            Review: 
                                                            {query}
                                                            Score:""")}
            ]
        
    elif dataset_num == 4:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of article-title pairs of a user. 
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, generate a title for the given article by the same author. Only output the title and nothing else.
                                                            Article: 
                                                            {query}
                                                            Title:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to generate a title for the given article. You will only output the title and nothing else.
                                                            Article: 
                                                            {query}
                                                            Title:""")}
            ]
        
    elif dataset_num == 5:   
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of abstract-title pairs of a scholar:
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, generate a title for the given abstract by the same author. Only output the title and nothing else.
                                                            Abstract:
                                                            {query}
                                                            Title:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to generate a title for the given abstract. You will only output the title and nothing else.
                                                            Abstract:
                                                            {query}
                                                            Title:""")}
            ]

    elif dataset_num == 7:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of tweets of a person:
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, paraphrase the given tweet by the same person. Only output the tweet and nothing else.
                                                            Tweet:
                                                            {query}
                                                            Paraphrased Tweet:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to paraphase a tweet. You will only output the title and nothing else.
                                                            Tweet:
                                                            {query}
                                                            Paraphrased Tweet:""")}
            ]          