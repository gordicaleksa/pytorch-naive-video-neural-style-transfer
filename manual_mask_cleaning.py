"""
    Automatic segmentation is not always perfect, this scripts helps by providing semi-automatic mask cleaning.
    Final option is using some editing software and editing masks there (high cost).

    Usual workflow:
        1. Copy processed_masks/ into processed_masks_refined/ (as this script is destructive)
        2. Manually inspect masks and find the range that can be filled/deleted with a rectangular/custom mask
        3. Tweak the params in top of the main function and run (start in debug mode if you're not sure how it works)
"""

import os
import sys
import enum
# Enables this project to see packages from pytorch-nst-feedforward submodule (e.g. utils)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch-nst-feedforward'))


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Using functions from utils package from pytorch-nst-feedforward submodule like load_image
from utils import utils


class Mode(enum.Enum):
    RECTANGULAR = 0,
    CUSTOM_MASK = 1


if __name__ == "__main__":
    # place your paths here
    mask_root = r'C:\tmp_data_dir\YouTube\CodingProjects\pytorch-naive-video-nst\data\clip_example\processed_masks_refined'
    clear_mask_path = r"C:\tmp_data_dir\YouTube\CodingProjects\pytorch-naive-video-nst\data\clip_example\custom_mask.png"
    mode = Mode.RECTANGULAR
    FIRST_IMAGE_INDEX = 0  # specify the first image in the directory that should be processed
    LAST_IMAGE_INDEX = 100  # and the last one (included)
    should_delete = False  # rectangular region and custom mask both fill (make white) the specified pixels

    if mode == Mode.CUSTOM_MASK:
        clear_mask = utils.load_image(clear_mask_path)[:, :, 0]

    for cnt, img_name in enumerate(os.listdir(mask_root)):
        img_path = os.path.join(mask_root, img_name)
        img = utils.load_image(img_path)[:, :, 0]

        # if in correct range
        if FIRST_IMAGE_INDEX <= cnt <= LAST_IMAGE_INDEX:
            # step1: edit image
            if mode == Mode.RECTANGULAR:
                # manually specify rectangular region here
                img[215:, 85:300] = 0. if should_delete else 1.
            elif mode == Mode.CUSTOM_MASK:
                img[clear_mask == 1.] = 0. if should_delete else 1.
            else:
                raise Exception(f'{mode} not supported.')
            # step2: overwrite old image
            cv.imwrite(img_path, np.uint8(img * 255))
