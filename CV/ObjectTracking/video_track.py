import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.sort import *
from utils.model import *
from utils.utils import *

# Step to download pre-train model
# import wget
# url = 'https://pjreddie.com/media/files/yolov3.weights'
# f_name = wget.download(url)
# print(f_name)

class track_video():
    def __init__(self,config_path,weights_path,class_path,conf_thres=0.8,nms_thres=0.4):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size=416
        self.model = Darknet(config_path, img_size=self.img_size)
        self.model.load_weights(weights_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.classes = load_classes(class_path)
    def detect_image(self,img):
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
             transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)),
             transforms.ToTensor(),
             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        if self.device == 'cpu':
            input_img = Variable(image_tensor.type(torch.FloatTensor))
        else:
            input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]      
        
        
    def run(self, videopath, save_path,sort_max_age=5,sort_min_hits=2,thicknss=2,frame_cons=1000):
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        vid = cv2.VideoCapture(videopath)
        
        mot_tracker = Sort(sort_max_age,sort_min_hits) 
        
        fps = vid.get(cv2.CAP_PROP_FPS)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        f_c = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        stp_cnt = min(f_c,frame_cons)
        for ii in tqdm(range(stp_cnt)):
            ret, frame = vid.read()
            if not ret:
                break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = self.detect_image(pilimg)

            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x
            if detections is not None:
                tracked_objects = mot_tracker.update(detections.cpu())

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cls = self.classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, thicknss)
        #             cv2.rectangle(frame, (x1, y1-20), (x1+len(cls)*19+60, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thicknss)

            vid_writer.write(frame)
        vid_writer.release()
        print('Video tracking result has been saved successfully at {}'.format(save_path))