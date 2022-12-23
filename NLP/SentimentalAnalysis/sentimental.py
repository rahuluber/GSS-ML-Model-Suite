from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import json

class SentimentAnalyser():
    
    def __init__(self, method='vader'):
        self.method = method
        if method=='vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif method=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        elif method=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
            self.model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
            self.model.save_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
            self.labels=['negative', 'neutral', 'positive']
        else:
            print('Model is not supported.')
            return None
        
    def run(self, text):
        if self.method=='vader':
            output = self.analyzer.polarity_scores(text)
        elif self.method=='bert':
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            score = np.argmax(scores)+1
            output = dict({'Start (Negative to Positive)':score})
        else:
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            output = dict({'negative':scores[0],'neutral':scores[1],'positive':scores[2]})
        return json.dumps(output)