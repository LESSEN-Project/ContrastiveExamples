def prepare_res_prompt(dataset, query, llm, examples=None, features=None):
    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompts()
    if features:
        features = "\n".join(features)
    context = llm.prepare_context(init_prompt, f"{query}\n{features}", examples) 
    return init_prompt.format(query=query, examples=context, features=features)

def prepare_summary_prompt(dataset, llm, examples, features):
    if dataset.name == "lamp":
        init_prompt = lamp_prompts(dataset.num, True)    
    if features:
        features = "\n".join(features)        
    context = llm.prepare_context(init_prompt, f"{features}", examples) 
    return init_prompt.format(examples=context, features=features)

def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())    

def amazon_prompts(step):
    if step == 1:
        return _ss_amazon_prompt()

def _ss_amazon_prompt() -> str:
    return strip_all("""You are an Amazon customer who writes reviews for the products you bought. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar product-reviews pairs from your past reviews:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulates sentence, compared to other writers:
                     <features>
                     {features}
                     </features>
                     Using the features, generate the proper review. If you haven't received any features besides similar pairs, only make use of them. 
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")

def lamp_prompts(dataset_num: int, summary=False) -> str:
    if summary:
        ss_lamp_prompts = {
            4: _get_lamp_feat_4,
            5: _get_lamp_feat_5,
            7: _get_lamp_feat_7
        }
    else:
        ss_lamp_prompts = {
            4: _ss_lamp_prompt_4,
            5: _ss_lamp_prompt_5,
            7: _ss_lamp_prompt_7
        }
    return ss_lamp_prompts.get(dataset_num)()

def _get_lamp_feat_4():
    return strip_all("""Your task is to highlight how a news editor formulates their titles for articles. 
                     You will receive a set of features and article-title pairs from the editor's past work to help you understand their style.
                     Here is the article-title pairs:
                     <pairs>
                     {examples}
                     </pairs>
                     Here are features:
                     <features>
                     {features}
                     </features>
                     Using the features and pairs, summarize the following concepts explaining the writing style of the editor: 
                     - How they use words
                     - How they formulate sentences
                     - Which words they use frequently
                     - Tone and sentiment
                     - Their sentence structure
                     You can include any other information that you think is important to capture the writing style of the editor.
                     Your output should be a bulleted list, each item explaining one crucial aspect about the writing style and how the editor constructs titles. 
                     Make the items concise, and don't include more than 5 items.
                     Do not output anything else.""")

def _get_lamp_feat_5():
    return strip_all("""Your task is to highlight how a scholar formulates their titles for abstracts. 
                     You will receive a set of features and abstract-title pairs from the scholar's past work to help you understand their style.
                     Here is the abstract-title pairs:
                     <pairs>
                     {examples}
                     </pairs>
                     Here are features:
                     <features>
                     {features}
                     </features>
                     Using the features and pairs, summarize the following concepts explaining the writing style of the scholar: 
                     - How they use words
                     - How they formulate sentences
                     - Which words they use frequently
                     - Tone and sentiment
                     - Their sentence structure
                     You can include any other information that you think is important to capture the writing style of the scholar.
                     Your output should be a bulleted list, each item explaining one crucial aspect about the writing style and how the scholar constructs titles. 
                     Make the items concise, and don't include more than 5 items.
                     Do not output anything else.""")

def _get_lamp_feat_7():
    return strip_all("""Your task is to highlight how a twitter user writes their tweets. 
                     You will receive a set of features and past tweets from the user to help you understand their style.
                     Here are the past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Here are features:
                     <features>
                     {features}
                     </features>
                     Using the features and past tweets, summarize the following concepts explaining the writing style of the user: 
                     - How they use words
                     - How they formulate sentences
                     - Which words they use frequently
                     - Tone and sentiment
                     - Their sentence structure
                     You can include any other information that you think is important to capture the writing style of the user.
                     Your output should be a bulleted list, each item explaining one crucial aspect about the writing style and how the user constructs tweets. 
                     Make the items concise, and don't include more than 5 items.
                     Do not output anything else.""")

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
                     Using the features, generate the proper title. If you haven't received any features besides similar pairs, only make use of them. 
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
                     Using the features, generate the proper title. If you haven't received any features besides similar pairs, only make use of them. 
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
                     Using the features, rephrase the tweet. If you haven't received any features besides past tweets, only make use of them. 
                     Only output the rephrased tweet and nothing else.
                     Tweet:
                     {query}
                     Paraphrased Tweet:""")