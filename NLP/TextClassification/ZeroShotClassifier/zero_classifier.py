from transformers import pipeline

class ZeroShot_Classifier():
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
        
    def run(self, sequence_to_classify, candidate_labels):
        output = self.classifier(sequence_to_classify, candidate_labels)
        return output

                      
