from torchvision import transforms
import numpy as np
import torch
from torchvision import models



class ImageSimilarity():
    def __init__(self):
        inputDim = (224,224)
        self.transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        
        # normalize the resized images as expected by resnet18
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def getVec(self, img):
        
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        
        return cnnModel, layer
    
    def run(self, img1, img2):
        resized1 = self.transformationForCNNInput(img1)
        resized2 = self.transformationForCNNInput(img2)
        
        vec1 = self.getVec(resized1)
        vec2 = self.getVec(resized2)
        
        similarity = sim = np.inner(vec1.T, vec2.T) / ((np.linalg.norm(vec1, axis=0).reshape(-1,1)) * ((np.linalg.norm(vec2, axis=0).reshape(-1,1)).T))[0][0]
        
        return similarity


