import os


import cv2 as cv
import numpy as np


from .constants import *


# Using functions from utils package from pytorch-nst-feedforward submodule like load_image
from utils import utils


def stylized_frames_mask_combiner(relevant_directories, dump_frame_extension, other_style=None):
    # in dirs
    frames_dir = relevant_directories['frames_path']
    mask_frames_dir = relevant_directories['processed_masks_dump_path']
    stylized_frames_dir = relevant_directories['stylized_frames_path']

    # out dirs (we'll dump combined imagery here)
    dump_path = os.path.join(stylized_frames_dir, os.path.pardir)
    model_name_suffix = '_' + os.path.basename(os.path.split(other_style)[0]) if other_style is not None else ''
    dump_path_bkg_masked = os.path.join(dump_path, 'composed_background_masked' + model_name_suffix)
    dump_path_person_masked = os.path.join(dump_path, 'composed_person_masked' + model_name_suffix)
    os.makedirs(dump_path_bkg_masked, exist_ok=True)
    os.makedirs(dump_path_person_masked, exist_ok=True)

    # if other_stylized_frames_path exists overlay frames are differently styled frames and not original frames
    if other_style is not None:
        overlay_frames_dir = other_style
    else:
        overlay_frames_dir = frames_dir

    if len(os.listdir(dump_path_bkg_masked)) == 0 and len(os.listdir(dump_path_person_masked)) == 0:
        for cnt, (name1, name2, name3) in enumerate(zip(sorted(os.listdir(stylized_frames_dir)), sorted(os.listdir(mask_frames_dir)), sorted(os.listdir(overlay_frames_dir)))):
            s_img_path = os.path.join(stylized_frames_dir, name1)  # stylized original frame image
            m_img_path = os.path.join(mask_frames_dir, name2)  # mask image
            o_img_path = os.path.join(overlay_frames_dir, name3)  # overlay image

            # load input imagery
            s_img = utils.load_image(s_img_path)
            m_img = utils.load_image(m_img_path, target_shape=s_img.shape[:2])
            o_img = utils.load_image(o_img_path, target_shape=s_img.shape[:2])

            # prepare canvas imagery
            combined_img_background = s_img.copy()
            combined_img_person = s_img.copy()

            # create masks
            background_mask = m_img == 0.
            person_mask = m_img == 1.

            # apply masks
            combined_img_background[background_mask] = o_img[background_mask]
            combined_img_person[person_mask] = o_img[person_mask]

            # save combined imagery
            combined_img_background_path = os.path.join(dump_path_bkg_masked, str(cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            combined_img_person_path = os.path.join(dump_path_person_masked, str(cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            cv.imwrite(combined_img_background_path, (combined_img_background * 255).astype(np.uint8)[:, :, ::-1])
            cv.imwrite(combined_img_person_path, (combined_img_person * 255).astype(np.uint8)[:, :, ::-1])
    else:
        print('Skipping combining with masks, already done.')

    return {"dump_path_bkg_masked": dump_path_bkg_masked, "dump_path_person_masked": dump_path_person_masked}