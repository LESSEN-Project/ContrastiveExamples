def prepare_prompt(dataset, query, llm, examples, features=None):
    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompts()
    context = llm.prepare_context(init_prompt, f"{query}\n{features}", examples)
    if features:
        features = "\n".join(features)
    return init_prompt.format(query=query, examples=context, features=features)

def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())

def amazon_prompts():
    return _RAG_amazon_prompt()

def _RAG_amazon_prompt() -> str:
    return strip_all("""You are an Amazon customer who writes reviews for the products you bought. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar product-reviews pairs from your past reviews:
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Now you will receive features shedding light into how you use words and formulates sentence, compared to other writers:
                     {features}
                     Using the features, generate the proper review. If you haven't received any features besides similar pairs, only make use of them. 
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")

def lamp_prompts(dataset_num: int) -> str:
    RAG_lamp_prompts = {
        4: _RAG_lamp_prompt_4,
        5: _RAG_lamp_prompt_5,
        7: _RAG_lamp_prompt_7
    }
    return RAG_lamp_prompts.get(dataset_num)()

def _RAG_lamp_prompt_4() -> str:
    return strip_all("""You are a news editor that generates titles for articles. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar article-title pairs from your past works:
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Now you will receive features shedding light into how you use words and formulates sentence, compared to other writers:
                     {features}
                     Using the features, generate the proper title. If you haven't received any features besides similar pairs, only make use of them. 
                     Only output the title and nothing else.
                     Article: 
                     {query}
                     Title:""")

def _RAG_lamp_prompt_5() -> str:
    return strip_all("""You are a scholar that generates titles for abstracts. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar abstract-title pairs from your past works:
                     <SimilarPairs>
                     {examples}
                     </SimilarPairs>
                     Now you will receive features shedding light into how you use words and formulates sentence, compared to other writers:
                     {features}
                     Using the features, generate the proper title. If you haven't received any features besides similar pairs, only make use of them. 
                     Only output the title and nothing else.
                     Abstract:
                     {query}
                     Title:""")

def _RAG_lamp_prompt_7() -> str:
    return strip_all("""You are a Twitter user who rephrases their own tweets. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar tweets from your past ones:
                     <SimilarTweets>
                     {examples}
                     </SimilarTweets>
                     Now you will receive features shedding light into how you use words and formulates sentence, compared to other twiiter users:
                     {features}
                     Using the features, rephrase the tweet. If you haven't received any features besides similar tweets, only make use of them. 
                     Only output the rephrased tweet and nothing else.
                     Tweet:
                     {query}
                     Paraphrased Tweet:""")