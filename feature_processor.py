from typing import Counter
import os
import json

import numpy as np
from textblob import TextBlob
import textstat
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer

class FeatureProcessor():

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

    def analyze_pos_usage(self, text):

        words = word_tokenize(text)

        pos_tags = pos_tag(words)

        adverbs = [word for word, pos in pos_tags if pos.startswith('RB')]
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        pronouns = [word for word, pos in pos_tags if pos.startswith('PRP')]

        total_words = len(words)

        adverbs_percentage = round((len(adverbs) / total_words) * 100, 2)
        adjectives_percentage = round((len(adjectives) / total_words) * 100, 2)
        pronouns_percentage = round((len(pronouns) / total_words) * 100, 2)

        return {
            'adverbs': {
                'count': len(adverbs),
                'percentage': adverbs_percentage,
            },
            'adjectives': {
                'count': len(adjectives),
                'percentage': adjectives_percentage,
            },
            'pronouns': {
                'count': len(pronouns),
                'percentage': pronouns_percentage,
            },
            'total_words': total_words
        }

    def get_word_frequency(self, text):

        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        total_words = sum(word_freq.values())

        word_freq_percentage = {word: round((count / total_words) * 100, 3) for word, count in word_freq.items()} if total_words > 0 else {} 
        sorted_word_freq_percentage = dict(sorted(word_freq_percentage.items(), key=lambda item: item[1], reverse=True))   
        return sorted_word_freq_percentage
    
    def feat_name_mappings(self):
        return {
            "MSWC": "Mean Sentence Word Count",
            "SP": "Sentiment Polarity"
        }

    def get_feat_file(self, dataset):
        os.makedirs("features", exist_ok=True)
        file_path = os.path.join("features", f"{dataset}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return dict()

    def save_feat_file(self, dataset, obj):
        file_path = os.path.join("features", f"{dataset}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_auth_features(self, dataset, feature_list, retr_gts, retr_texts=None):
        if retr_texts:
            author_texts = []
            for retr_text, retr_gt in zip(retr_texts, retr_gts): 
                author_texts.append([f"{gt}: {text}" for text, gt in zip(retr_text, retr_gt)])
        else:
            author_texts = retr_gts
        author_features = self.get_feat_file(dataset)
        for feature in feature_list:
            if feature not in author_texts.keys():
                if feature == "MSWC":
                    author_features[feature] = [self.feature_processor.get_average_sentence_length(text) for text in author_texts]
                elif feature == "SP":
                    author_features[feature] = [self.feature_processor.get_sentiment_polarity(text) for text in author_texts]
            self.save_feat_file(dataset, author_features)
        return author_features
    
    def get_features(self, dataset, feature_list, retr_gts, retr_texts=None):
        author_features = self.get_auth_features(dataset, feature_list, retr_texts, retr_gts)
        all_author_features = []
        mean_features = dict()
        std_features = dict()
        for author in author_features:
            proc_author_features = []
            for feature in feature_list:
                if feature != "word_frequencies":
                    if feature not in mean_features.keys():
                        mean_features[feature] = np.mean([author[feature] for author in author_features])
                        std_features[feature] = np.std([author[feature] for author in author_features])
                    mean_value = round(mean_features[feature], 3)      
                    pers_value = round(author[feature], 3)
                    std_value = round(abs(pers_value-mean_value)/std_features[feature], 3) 
                    feat_desc = f"{self.feat_name_mappings[feature]} for the writer: {pers_value}, mean value for all the writers: {mean_value}, which makes it {std_value} standard deviations away."
                else:
                    most_used_words = list(author[feature].items())[:10]
                    feat_desc = f"The list of most frequently used words for the writer: {most_used_words}"
                proc_author_features.append(feat_desc)
            all_author_features.append(proc_author_features)
        return all_author_features