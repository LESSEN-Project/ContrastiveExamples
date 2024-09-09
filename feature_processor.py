from typing import Counter
import os
import json

import numpy as np
from textblob import TextBlob
import textstat
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

class FeatureProcessor():

    def __init__(self) -> None:
        nltk.download('punkt', quiet=True)      
        nltk.download('stopwords', quiet=True)  
        nltk.download('averaged_perceptron_tagger', quiet=True)

    def get_vocabulary_richness(self, texts):
        all_texts = " ".join(texts)
        words = word_tokenize(all_texts.lower())
        return len(set(words)) / len(words)

    def get_average_sentence_length(self, texts):
        return np.mean([len(text.split(" ")) for text in texts])

    def get_sentiment_polarity(self, texts):
        return np.mean([TextBlob(text).sentiment.polarity for text in texts])

    def get_subjectivity(self, texts):
        return np.mean([TextBlob(text).sentiment.subjectivity for text in texts])

    def get_passive_voice_count(self, texts):
        pass_count = []
        nlp = spacy.load('en_core_web_sm')
        for text in texts:
            doc = nlp(text)
            pass_count.append(sum(1 for token in doc if token.dep_ == "nsubjpass"))
        return np.mean(pass_count)
                            
    def get_readability_score(self, texts):
        return np.mean([textstat.flesch_kincaid_grade(text) for text in texts])
    
    def get_adverb_usage(self, texts):

        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        pos_tags = pos_tag(words)

        adverbs = [word for word, pos in pos_tags if pos.startswith('RB')]
        return round((len(adverbs) / len(words)) * 100, 2)
    
    def get_adjective_usage(self, texts):

        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        pos_tags = pos_tag(words)

        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        return round((len(adjectives) / len(words)) * 100, 2)
    
    def get_pronoun_usage(self, texts):

        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        pos_tags = pos_tag(words)

        pronouns = [word for word, pos in pos_tags if pos.startswith('PRP')]
        return round((len(pronouns) / len(words)) * 100, 2)

    def get_word_frequency(self, texts):

    
        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        total_words = sum(word_freq.values())

        word_freq_percentage = {word: round((count / total_words) * 100, 3) for word, count in word_freq.items()} if total_words > 0 else {} 
        sorted_word_freq_percentage = dict(sorted(word_freq_percentage.items(), key=lambda item: item[1], reverse=True))   
        return sorted_word_freq_percentage
    
    def feat_name_mappings(self):
        return {
            "MSWC": {
                "full_name": "Mean Sentence Word Count",
                "desc": "The number of words the writer uses in a sentence on average"
            },
            "SP": {
                "full_name": "Sentiment Polarity",
                "desc": "Average sentiment polarity for the writer (between -1 to 1)"
            },
            "WF": {
                "full_name": "Word Frequencies",
                "desc": "Frequency of words the writer uses in percentages"
            },
            "ADVU": {
                "full_name": "Average Adverb Usage Percentage",
                "desc": "The percentage (between 0-100) of adverbs the writer uses on average"  
            },
            "ADJU": {
                "full_name": "Average Adjective Usage Percentage",
                "desc": "The percentage (between 0-100) of adjectives the writer uses on average"                  
            },
            "PU": {
                "full_name": "Average Pronoun Usage Percentage",
                "desc": "The percentage (between 0-100) of pronouns the writer uses on average"                  
            }
        }

    def get_feat_file(self, dataset):
        os.makedirs("features", exist_ok=True)
        file_path = os.path.join("features", f"{dataset}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return dict()

    def save_feat_file(self, file_name, obj):
        file_path = os.path.join("features", f"{file_name}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_auth_features(self, dataset, feature_list, retr_gts, retr_texts=None):
        if retr_texts:
            file_name = f"{dataset}_texts"
            author_texts = []
            for retr_text, retr_gt in zip(retr_texts, retr_gts): 
                author_texts.append([f"{gt}: {text}" for text, gt in zip(retr_text, retr_gt)])
        else:
            file_name = f"{dataset}_no_texts"
            author_texts = retr_gts
        author_features = self.get_feat_file(dataset)
        for feature in feature_list:
            if feature not in author_features.keys():
                if feature == "MSWC":
                    author_features[feature] = [self.get_average_sentence_length(text) for text in author_texts]
                elif feature == "SP":
                    author_features[feature] = [self.get_sentiment_polarity(text) for text in author_texts]
                elif feature == "WF":
                    author_features[feature] = [self.get_word_frequency(text) for text in author_texts]
                elif feature == "ADVU":
                    author_features[feature] = [self.get_adverb_usage(text) for text in author_texts]
                elif feature == "ADJU":
                    author_features[feature] = [self.get_adjective_usage(text) for text in author_texts]
                elif feature == "PU":
                    author_features[feature] = [self.get_pronoun_usage(text) for text in author_texts]
            self.save_feat_file(file_name, author_features)
        return author_features
    
    def get_features(self, dataset, feature_list, retr_gts, retr_texts=None):
        author_features = self.get_auth_features(dataset, feature_list, retr_gts, retr_texts)
        all_author_features = []
        mean_features = dict()
        std_features = dict()
        for i in range(len(retr_gts)):
            proc_author_features = []
            for feature in feature_list:
                if feature != "WF":
                    if feature not in mean_features.keys():
                        mean_features[feature] = np.mean(author_features[feature])
                        std_features[feature] = np.std(author_features[feature])
                    mean_value = round(mean_features[feature], 3)      
                    pers_value = round(author_features[feature][i], 3)
                    std_value = round(abs(pers_value-mean_value)/std_features[feature], 3) 
                    feat_desc = f"-{self.feat_name_mappings()[feature]['desc']} is {pers_value}, mean value for all the writers is {mean_value}, which makes it {std_value} standard deviations away from the mean."
                else:
                    most_used_words = [f"{w}: {p}" for w, p in list(author_features[feature][i].items())[:10]]
                    feat_desc = f"-The list of most frequently used words of the writer, alongside the percentages they appear in writer's dictionary: {most_used_words}"
                proc_author_features.append(feat_desc)
            all_author_features.append(proc_author_features)
        return all_author_features