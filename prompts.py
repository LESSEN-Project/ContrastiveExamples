def prepare_res_prompt(dataset, query, llm, examples=None, features=None, counter_examples=None):

    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompts()
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

def amazon_prompts():
    return _ss_amazon_prompt()

def _ss_amazon_prompt() -> str:
    return strip_all("""You are an Amazon customer who writes reviews for the products you buy. You will be provided a set of features to help you understand your writing style.
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

def lamp_prompts(dataset_num: int) -> str:

    ss_lamp_prompts = {
            4: _ss_lamp_prompt_4,
            5: _ss_lamp_prompt_5,
            7: _ss_lamp_prompt_7
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

def _ss_lamp_prompt_7() -> str:
    return strip_all("""You are a Twitter user who wants to rephrase their own tweets. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is your past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive tweets from other users to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Using the features, rephrase the tweet. If you haven't received some of the features, only make use of the provided ones.
                     Only output the rephrased tweet and nothing else.
                     Tweet:
                     {query}
                     Paraphrased Tweet:""")