import numpy as np


FILE_NAME_NUM_DIGITS = 6  # number of digits in the frame/mask names, e.g. for 6: '000023.jpg'
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4']

PERSON_CHANNEL_INDEX = 15  # segmentation stage

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CUDA_EXCEPTION_CODE = 1
ERROR_CODE = 1