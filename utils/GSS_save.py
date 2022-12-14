import os
import cv2 as cv
import numpy as np
import pandas as pd

def image_save_size(I, img_name, img_size):
    for qlt in np.arange(95,10,-5):
            cv.imwrite(img_name, I, [int(cv.IMWRITE_JPEG_QUALITY), qlt])
            if os.stat(img_name).st_size < img_size:
                break

def image_save(I, img_name):
    cv.imwrite(img_name, I)

