from transformers import pipeline

class AbstractiveSummary():
    def __init__(self, max_length=130,min_length=30):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.max_length = max_length
        self.min_length = min_length
        
    def run(self, text):
        output = self.summarizer(text, max_length=self.max_length, min_length=self.min_length, do_sample=False)
        return output