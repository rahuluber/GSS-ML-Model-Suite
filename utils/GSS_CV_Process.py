import cv2 as cv
import numpy as np


def resize_image_pad(I, min_h, min_w,asp_ratio,clr=[255,255,255]):
    h,w,c = I.shape
    if w>=(asp_ratio*h):
        if w<min_w:
            I_resized = I
        else:
            rs = min_w/w
            w_new = int(np.ceil(w*rs))
            h_new = int(np.ceil(h*rs))
            I_resized = cv.resize(I,(w_new, h_new), interpolation = cv.INTER_CUBIC)
    else:
        if h<min_h:
            I_resized = I
        else:
            rs = min_h/h
            w_new = int(np.ceil(w*rs))
            h_new = int(np.ceil(h*rs))
            I_resized = cv.resize(I,(w_new, h_new), interpolation = cv.INTER_CUBIC)
    I_new = np.ones((min_h,min_w,3)) #*255
    pad_clr = np.reshape(clr,(1,1,3))
    I_new = np.multiply(I_new,pad_clr)
    I_new[int((min_h/2)-np.ceil(h_new/2)):int((min_h/2)+np.floor(h_new/2)),int((min_w/2)-np.ceil(w_new/2)):int((min_w/2)+np.floor(w_new/2)),:] = I_resized
    return I_new.astype(np.uint8)

def resize_image_hw(I, h, w):
    I_resized = cv.resize(I,(w, h), interpolation = cv.INTER_CUBIC)
    return I_resized


def resize_image_aratio(I, min_h, min_w,asp_ratio):
    h,w,c = I.shape
    if w>=(asp_ratio*h):
        if w<min_w:
            I_resized = I
        else:
            rs = min_w/w
            w_new = int(np.ceil(w*rs))
            h_new = int(np.ceil(h*rs))
            I_resized = cv.resize(I,(w_new, h_new), interpolation = cv.INTER_CUBIC)
    else:
        if h<min_h:
            I_resized = I
        else:
            rs = min_h/h
            w_new = int(np.ceil(w*rs))
            h_new = int(np.ceil(h*rs))
            I_resized = cv.resize(I,(w_new, h_new), interpolation = cv.INTER_CUBIC)
    
    return I_resized



