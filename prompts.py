def prepare_improvement_prompt(dataset, query, llm, examples, features, initial_output):

    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num, True)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompts(True)
    features = "\n".join(features)
    context = llm.prepare_context(init_prompt, f"{query}\n{features}", examples) 

    return init_prompt.format(query=query, examples=context, features=features, initial_output=initial_output)

def prepare_res_prompt(dataset, query, llm, examples, features=None, counter_examples=None):

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

def amazon_prompts(two_step=False):
    if two_step:
        return _amazon_fcheck_prompt()
    else:
        return _ss_amazon_prompt()

def _amazon_fcheck_prompt():
    return strip_all("""You are an Amazon customer that likes to write reviews for products. An AI model has generated a review in your style for the last product you purchased.
                     You will receive a set of features so that you can check if its generation resembles your style.
                     First feature you will receive is similar product-review pairs from your past reviews:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive the product name and the review the AI model generated:
                     Product:
                     {query}
                     AI Review:
                     {initial_output}
                     Using the features, check how much the AI model followed your style. Generate an improved review where you follow the provided features better than the AI model.
                     Only output the review and nothing else.
                     Review:""")

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

def lamp_prompts(dataset_num: int, two_step=False) -> str:
    if two_step:
        fcheck_lamp_prompts = {
            4: _fcheck_lamp_prompt_4,
            5: _fcheck_lamp_prompt_5,
            7: _fcheck_lamp_prompt_7
        }
        return fcheck_lamp_prompts.get(dataset_num)()
    else:
        ss_lamp_prompts = {
            4: _ss_lamp_prompt_4,
            5: _ss_lamp_prompt_5,
            7: _ss_lamp_prompt_7
        }
        return ss_lamp_prompts.get(dataset_num)()

def _fcheck_lamp_prompt_4() -> str:
    return strip_all("""You are a news editor that generates titles for articles. An AI model has generated a title in your style for the last article you received.
                     You will receive a set of features so that you can check if its generation resembles your style.
                     First feature you will receive is similar article-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive the article and the title the AI model generated:
                     Article:
                     {query}
                     AI Title:
                     {initial_output}
                     Using the features, check how much the AI model followed your style. Generate an improved title where you follow the provided features better than the AI model.
                     Only output the title and nothing else.
                     Title:""")

def _fcheck_lamp_prompt_5() -> str:
    return strip_all("""You are a scholar that generates titles for abstracts. An AI model has generated a title in your style for the last abstract you received.
                     You will receive a set of features so that you can check if its generation resembles your style.
                     First feature you will receive is similar abstract-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive the abstract and the title the AI model generated:
                     Abstract:
                     {query}
                     AI Title:
                     {initial_output}
                     Using the features, check how much the AI model followed your style. Generate an improved title where you follow the provided features better than the AI model.
                     Only output the title and nothing else.
                     Title:""")

def _fcheck_lamp_prompt_7() -> str:
    return strip_all("""You are a Twitter user. An AI model has generated a rephrased tweet in your style for your last tweet.
                     You will receive a set of features so that you can check if its generation resembles your style.
                     First feature you will receive is your past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive your original tweet and the rephrased tweet the AI model generated:
                     Original Tweet:
                     {query}
                     AI Rephrased Tweet:
                     {initial_output}
                     Using the features, check how much the AI model followed your style. Generate an improved rephrased tweet where you follow the provided features better than the AI model.
                     Only output the rephrased tweet and nothing else.
                     Rephrased Tweet:""")

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