def prepare_res_prompt(dataset, query, llm, examples, features=None, counter_examples=None, use_cot_prompt=False):

    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num, use_cot_prompt)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompts(use_cot_prompt)
    
    if features:
        features = "\n".join(features)
    
    context = llm.prepare_context(init_prompt, f"{query}\n{features}", examples) 
    ce_examples = ""

    if counter_examples:
        i = 0
        for ce_example in counter_examples:
            ce_context = llm.prepare_context(init_prompt, f"{query}\n{features}\n{context}", ce_example) 
            if context:
                i += 1
                ce_examples = f"{ce_examples}\n<Other Writer-{i}>\n{ce_context}\n</Other Writer-{i}>\n"

    return init_prompt.format(query=query, examples=context, features=features, counter_examples=ce_examples)


def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())    


def amazon_prompts(use_cot_prompt=False):
    return _ss_amazon_prompt_cot() if use_cot_prompt else _ss_amazon_prompt()


def _ss_amazon_prompt() -> str:
    return strip_all("""You are an Amazon customer that likes to write reviews for products. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar product-review pairs from your past reviews to remind you of your style:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive product-review pairs from other customers to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Using the features, generate the proper review. If you haven't received some of the features, only make use of the provided ones. 
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")


def _ss_amazon_prompt_cot() -> str:
    return strip_all("""You are an Amazon customer that likes to write reviews for products. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar product-review pairs from your past reviews to remind you of your style:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive product-review pairs from other customers to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive the product:
                     {query}
                     Think step by step about the product, the features provided, and how they relate to your writing style before generating the review. If you haven't received some of the features, only make use of the provided ones.
                     After your thoughts, provide the review in a new line.""")


def lamp_prompts(dataset_num: int, use_cot_prompt=False) -> str:
    ss_lamp_prompts = {
        4: _ss_lamp_prompt_4_cot if use_cot_prompt else _ss_lamp_prompt_4,
        5: _ss_lamp_prompt_5_cot if use_cot_prompt else _ss_lamp_prompt_5,
        7: _ss_lamp_prompt_7_cot if use_cot_prompt else _ss_lamp_prompt_7
    }
    return ss_lamp_prompts.get(dataset_num)()


def _ss_lamp_prompt_4() -> str:
    return strip_all("""You are a news editor that generates titles for articles. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar article-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive article-title pairs from other editors to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Using the features, generate the proper title. If you haven't received some of the features, only make use of the provided ones. 
                     Only output the title and nothing else.
                     Article: 
                     {query}
                     Title:""")


def _ss_lamp_prompt_4_cot() -> str:
    return strip_all("""You are a news editor that generates titles for articles. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar article-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive article-title pairs from other editors to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive the article: 
                     {query}
                     Think step by step about the features and how you normally construct article titles before generating the proper title. If you haven't received some of the features, only make use of the provided ones. 
                     After your thoughts, provide the title at the end in a new line.""")


def _ss_lamp_prompt_5() -> str:
    return strip_all("""You are a scholar that generates titles for abstracts. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar abstract-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive abstract-title pairs from other scholars to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Using the features, generate the proper title. If you haven't received some of the features, only make use of the provided ones. 
                     Only output the title and nothing else.
                     Abstract:
                     {query}
                     Title:""")


def _ss_lamp_prompt_5_cot() -> str:
    return strip_all("""You are a scholar that generates titles for abstracts. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar abstract-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive abstract-title pairs from other scholars to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive the abstract:
                     {query}
                     Think step by step about the features and how you normally construct abstract titles before generating the proper title. If you haven't received some of the features, only make use of the provided ones. 
                     After your thoughts, provide the title at the end in a new line.""")


def _ss_lamp_prompt_7() -> str:
    return strip_all("""You are a Twitter user. Here is a set of your past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Here are some features about your writing style:
                     <features>
                     {features}
                     </features>
                     Finally, here are some tweets from other users:
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive your last tweet:
                     Tweet:
                     {query}
                     Using the provided information, rephrase the tweet so it better reflects your writing style. If you haven't received some of the information, only make use of the provided ones.
                     Only output the rephrased tweet and nothing else.
                     Rephrased Tweet:""")


def _ss_lamp_prompt_7_cot() -> str:
    return strip_all("""You are a Twitter user. Here is a set of your past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Here are some features about your writing style:
                     <features>
                     {features}
                     </features>
                     Finally, here are some tweets from other users:
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive your last tweet:
                     Tweet:
                     {query}
                     Think step by step about how your previous tweets were formulated and the features provided before rephrasing the tweet. If you haven't received some of the information, only make use of the provided ones.
                     After your thoughts, provide the rephrased tweet at the end in a new line.""")