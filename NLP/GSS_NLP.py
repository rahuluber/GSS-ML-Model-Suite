from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect

def lang_detection(text):
    return detect(text)

class SentenceSimilarity():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
    def run(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1, sentence2])
        output = cosine_similarity(embeddings1, embeddings1)[0,1]
        return output

class AbstractiveSummary():
    def __init__(self, max_length=130,min_length=30):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.max_length = max_length
        self.min_length = min_length
        
    def run(self, text):
        output = self.summarizer(text, max_length=self.max_length, min_length=self.min_length, do_sample=False)
        return output

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