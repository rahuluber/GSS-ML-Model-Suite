from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id
import torch
from PIL import Image
import io
import numpy

class segment_image():
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        self.model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
        
    def run(self, image):
        # prepare image for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # forward pass
        outputs = self.model(**inputs)

        # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = self.feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

        # the segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
        # retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb_to_id(panoptic_seg)
        return panoptic_seg, panoptic_seg_id
