def strip_all(text):
   return "\n".join([line.strip() for line in text.splitlines()])

def lamp_prompt(dataset, prof_text, examples=None):
    if dataset == 1:
            if examples:
                return [
                    {"role": "user", "content": strip_all(f"""Here are a couple of abstract-title pairs of a scholar.
                                                                <EXAMPLES>
                                                                {examples}
                                                                </EXAMPLES>
                                                                With the given examples, complete the following task. Only output the response of the task and nothing else.
                                                                Task:
                                                                {prof_text}
                                                                """)}
                ]
            else:
                return [
                    {"role": "user", "content": strip_all(f"""Complete the following task. Only output the response of the task and nothing else.
                                                                Task:
                                                                {prof_text}
                                                                """)}
                ]
            
    elif dataset == 2:
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
                                                            {prof_text}
                                                            Tag:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Choose the correct category tag for the following movie description between these tags:
                                                            [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]
                                                            Only output the tag and nothing else.
                                                            Description:
                                                            {prof_text}
                                                            Tag:""")}
            ]
        
    elif dataset == 3:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of review-rating pairs of a user. 
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, give a score between [1, 2, 3, 4, 5] to the following review by the same user. Only output the score and nothing else.
                                                            Review: 
                                                            {prof_text}
                                                            Score:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Give a score between [1, 2, 3, 4, 5] to the following review. Only output the score and nothing else.
                                                            Review: 
                                                            {prof_text}
                                                            Score:""")}
            ]
        
    elif dataset == 4:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of article-title pairs of a user. 
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, generate a title for the given article by the same author. Only output the title and nothing else.
                                                            Article: 
                                                            {prof_text}
                                                            Title:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to generate a title for the given article. You will only output the title and nothing else.
                                                            Article: 
                                                            {prof_text}
                                                            Title:""")}
            ]
        
    elif dataset == 5:   
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of abstract-title pairs of a scholar:
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, generate a title for the given abstract by the same author. Only output the title and nothing else.
                                                            Abstract:
                                                            {prof_text}
                                                            Title:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to generate a title for the given abstract. You will only output the title and nothing else.
                                                            Abstract:
                                                            {prof_text}
                                                            Title:""")}
            ]

    elif dataset == 7:
        if examples:
            return [
                {"role": "user", "content": strip_all(f"""Here are a couple of tweets of a person:
                                                            <EXAMPLES>
                                                            {examples}
                                                            </EXAMPLES>
                                                            With the given examples, paraphrase the given tweet by the same person. Only output the tweet and nothing else.
                                                            Tweet:
                                                            {prof_text}
                                                            Paraphrased Tweet:""")}
            ]
        else:
            return [
                {"role": "user", "content": strip_all(f"""Your task is to paraphase a tweet. You will only output the title and nothing else.
                                                            Tweet:
                                                            {prof_text}
                                                            Paraphrased Tweet:""")}
            ]          