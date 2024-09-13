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
        self.nlp = spacy.load("en_core_web_sm")

    def get_vocabulary_richness(self, texts):
        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        return round((len(set(words)) / len(words))*100, 2)

    def get_average_sentence_length(self, texts):
        return np.mean([len(text.split(" ")) for text in texts])

    def get_sentiment_polarity(self, texts):
        return np.mean([TextBlob(text).sentiment.polarity for text in texts])

    def get_subjectivity(self, texts):
        return np.mean([TextBlob(text).sentiment.subjectivity for text in texts])
    
    def get_smog_index(self, texts):
        return np.mean([textstat.smog_index(text) for text in texts])
    
    def get_passive_voice_usage(self, texts):

        texts = ".".join(texts)
        doc = self.nlp(texts)
        
        total_sentences = 0
        passive_sentences = 0
        
        for sent in doc.sents:
            total_sentences += 1
            if any(token.dep_ == "nsubjpass" for token in sent):
                passive_sentences += 1
        
        passive_percentage = (passive_sentences / total_sentences) * 100 if total_sentences > 0 else 0
        return round(passive_percentage, 2)
                            
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
        sorted_word_freq_percentage = list(sorted(word_freq_percentage.items(), key=lambda item: item[1], reverse=True))   
        return sorted_word_freq_percentage

    def get_named_entity_freqency(self, texts):

        texts = " ".join(texts)
        doc = self.nlp(texts)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        entity_counter = Counter(named_entities)
        total_entities = sum(entity_counter.values())
        
        sorted_entities = [
            ((ent, label), round((count / total_entities) * 100, 2))
            for (ent, label), count in entity_counter.most_common()
        ]
        
        return sorted_entities
    
    def get_dep_pattern_frequency(self, texts):

        texts = " ".join(texts)
        doc = self.nlp(texts)
        
        dependency_patterns = [(token.text, token.dep_) for token in doc]
        pattern_counter = Counter(dependency_patterns)
        total_patterns = sum(pattern_counter.values())
        
        sorted_patterns = [
            (pattern, round((count / total_patterns) * 100, 2))
            for pattern, count in pattern_counter.most_common()
        ]
        return sorted_patterns
    
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
                "desc": "The list of most frequently used words of the writer, alongside the percentages they appear in writer's dictionary"
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
            },
            "SUBJ": {
                "full_name": "Subjectivity",
                "desc": "Average subjectivity for the writer (between 0 to 1, 1 being the most subjective)"
            },
            "PASSU": {
                "full_name": "Average Passive Voice Usage Percentage",
                "desc": "The percentage (between 0-100) of sentences the writer forms in passive voice on average"
            },
            "NEF": {
                "full_name": "Named Entity Frequencies",
                "desc": "The list of most frequently used named entities of the writer, alongside the percentages they appear in writer's dictionary"
            },
            "DPF": {
                "full_name": "Dependency Pattern Frequencies",
                "desc": "The list of most frequently used dependency patterns of the writer, alongside the percentages they appear in writer's dictionary"
            },
            "VR": {
                "full_name": "Vocabulary Richness",
                "desc": "The percentage (between 0-100) of the vocabulary richness of the writer, a high number meaning a richer vocabulary"                
            },
            "SMOG": {
                "full_name": "SMOG Index",
                "desc": "Average SMOG Index of the writer, which measures how many years of education is needed to understand the text"
            }
        }

    def get_feat_file(self, file_name):
        os.makedirs("features", exist_ok=True)
        file_path = os.path.join("features", f"{file_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return dict()

    def save_feat_file(self, file_name, obj):
        file_path = os.path.join("features", f"{file_name}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_auth_features(self, dataset, feature_list, retr_texts, retr_gts):

        file_name = f"{dataset}_feats"
        author_texts = retr_gts
        if retr_texts != retr_gts:
            full_auth_texts = retr_texts
        else:
            full_auth_texts = author_texts
        author_features = self.get_feat_file(file_name)

        for feature in feature_list:
            if feature not in author_features.keys():
                print(f"Preparing {feature}")
                if feature == "MSWC":
                    author_features[feature] = [self.get_average_sentence_length(text) for text in author_texts]
                elif feature == "VR":
                    author_features[feature] = [self.get_vocabulary_richness(text) for text in author_texts]
                elif feature == "SP":
                    author_features[feature] = [self.get_sentiment_polarity(text) for text in author_texts]
                elif feature == "SMOG":
                    author_features[feature] = [self.get_smog_index(text) for text in author_texts]
                elif feature == "ADVU":
                    author_features[feature] = [self.get_adverb_usage(text) for text in author_texts]
                elif feature == "ADJU":
                    author_features[feature] = [self.get_adjective_usage(text) for text in author_texts]
                elif feature == "PU":
                    author_features[feature] = [self.get_pronoun_usage(text) for text in author_texts]
                elif feature == "SUBJ":
                    author_features[feature] = [self.get_subjectivity(text) for text in author_texts]
                elif feature == "PASSU":
                    author_features[feature] = [self.get_passive_voice_usage(text) for text in author_texts]
                elif feature == "WF":
                    author_features[feature] = [self.get_word_frequency(text) for text in author_texts]
                elif feature == "NEF":
                    author_features[feature] = [self.get_named_entity_freqency(text) for text in full_auth_texts]
                elif feature == "DPF":
                    author_features[feature] = [self.get_dep_pattern_frequency(text) for text in full_auth_texts]
            self.save_feat_file(file_name, author_features)
        return author_features
    
    def get_features(self, dataset, feature_list, retr_texts, retr_gts, top_k=10):
        author_features = self.get_auth_features(dataset, feature_list, retr_texts, retr_gts)
        all_author_features = []
        mean_features = dict()
        std_features = dict()
        for i in range(len(retr_gts)):
            proc_author_features = []
            for feature in feature_list:
                if not feature.endswith("F"):
                    if feature not in mean_features.keys():
                        mean_features[feature] = np.mean(author_features[feature])
                        std_features[feature] = np.std(author_features[feature])
                    mean_value = round(mean_features[feature], 3)      
                    pers_value = round(author_features[feature][i], 3)
                    std_value = round(abs(pers_value-mean_value)/std_features[feature], 3) 
                    feat_desc = f"-{self.feat_name_mappings()[feature]['desc']} is {pers_value}, mean value for all the writers is {mean_value}, which makes it {std_value} standard deviations away from the mean."
                else:
                    most_used_vals = [w for w, _ in author_features[feature][i][:top_k]]
                    feat_desc = f"-{self.feat_name_mappings()[feature]['desc']}: {most_used_vals}"
                proc_author_features.append(feat_desc)
            all_author_features.append(proc_author_features)
        return all_author_features