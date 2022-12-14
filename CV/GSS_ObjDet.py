from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from utils.GSS_visualization import *
import matplotlib.pyplot as plt
import json
import pandas as pd

class localize_object():
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50") 
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
    def run(self, image, threshold=0.9):
        inputs = self.feature_extractor(images=image, return_tensors="pt") 
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]) 
        results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
        obj_label=[]
        obj_bbox=[]
        obj_conf=[]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if score > threshold:
                obj_label.append(self.model.config.id2label[label.item()])
                obj_conf.append(round(score.item(), 3))
                obj_bbox.append(box)
                
        df = dict({'Label':obj_label, 'Conf_score':obj_conf,'bounding_box':obj_bbox})
        output = json.dumps(df)
        return output



