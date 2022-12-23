from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SentenceSimilarity():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
    def run(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1, sentence2])
        output = cosine_similarity(embeddings1, embeddings1)[0,1]
        return output