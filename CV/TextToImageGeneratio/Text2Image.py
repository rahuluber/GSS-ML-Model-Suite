from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

class ImageGenerator():
    
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
    def run(self, prompt):
        image = self.pipe(prompt).images[0]
        return image

