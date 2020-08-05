import argparse
import subprocess
import time
import os
import shutil
import sys
# Enables this file to see packages from pytorch-nst-feedforward submodule (e.g. utils)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch-nst-feedforward'))


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# Using functions from utils package from pytorch-nst-feedforward submodule like load_image
from utils import utils
from pipeline_components.segmentation import extract_person_masks_from_frames
from pipeline_components.video_creation import create_videos
from pipeline_components.constants import *

SUPPORTED_VIDEO_EXTENSIONS = ['.mp4']


def stylization(frames_path):
    model_name = 'mosaic_4e5_e2.pth'
    stylized_frames_dump_dir = os.path.join(frames_path, os.path.pardir, os.path.pardir, model_name.split('.')[0], 'stylized')

    if len(os.listdir(stylized_frames_dump_dir)) == 0:
        for frame_name in os.listdir(frames_path):
            frame_path = os.path.join(frames_path, frame_name)
            subprocess.call(['python', 'pytorch-nst-feedforward/stylization_script.py', '--content_img_name', frame_path, '--img_width', '500', '--model_name', model_name, '--redirected_output', stylized_frames_dump_dir])
    else:
        print('Skipping frame stylization, already done.')

    return {"stylized_frames_path": stylized_frames_dump_dir}


def stylized_frames_mask_combiner(relevant_directories, dump_frame_extension, other_stylized_frames_dir=None):
    print(f'Combining {processed_video_dir}.')

    # in dirs
    frames_dir = relevant_directories['frames_path']
    mask_frames_dir = relevant_directories['processed_masks_dump_path']
    stylized_frames_dir = relevant_directories['stylized_frames_path']

    # out dirs (we'll dump combined imagery here)
    dump_path = os.path.join(stylized_frames_dir, os.path.pardir)
    dump_path_bkg_masked = os.path.join(dump_path, 'composed_background_masked')
    dump_path_person_masked = os.path.join(dump_path, 'composed_person_masked')
    os.makedirs(dump_path_bkg_masked, exist_ok=True)
    os.makedirs(dump_path_person_masked, exist_ok=True)

    # if other_stylized_frames_path exists overlay frames are differently styled frames and not original frames
    if other_stylized_frames_dir is not None:
        overlay_frames_dir = other_stylized_frames_dir
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


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    # todo: figure out why is jpg giving me bad visual quality for dummy.mp4 video
    frame_extension = '.jpg'  # .jpg is suitable to use here - smaller size and unobservable quality loss
    mask_extension = '.png'  # don't use .jpg here! bigger size + corrupts the binary property of the mask when loaded
    frame_name_format = f'%0{FILE_NAME_NUM_DIGITS}d{frame_extension}'  # e.g. 000023.jpg
    data_path = os.path.join(os.path.dirname(__file__), 'data')

    #
    # Modifiable args
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_batch_size", type=int, help="Number of images in a batch", default=12)
    parser.add_argument("--delete_source_imagery", type=bool, help="Should delete imagery after videos are created", default=False)
    args = parser.parse_args()

    #
    # For every video located under data/
    #
    for element in os.listdir(data_path):
        maybe_video_path = os.path.join(data_path, element)
        if os.path.isfile(maybe_video_path) and os.path.splitext(maybe_video_path)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS:
            video_path = maybe_video_path
            video_name = os.path.basename(video_path).split('.')[0]

            processed_video_dir = os.path.join(data_path, 'clip_' + video_name)
            os.makedirs(processed_video_dir, exist_ok=True)

            frames_path = os.path.join(processed_video_dir, 'frames', 'frames')
            os.makedirs(frames_path, exist_ok=True)

            ffmpeg = 'ffmpeg'
            if shutil.which(ffmpeg):  # if ffmpeg is in system path
                cap = cv.VideoCapture(video_path)
                fps = int(cap.get(cv.CAP_PROP_FPS))

                out_frame_pattern = os.path.join(frames_path, frame_name_format)
                audio_dump_path = os.path.join(processed_video_dir, video_name + '.aac')

                #
                # step1: Extract frames from the videos as well as audio file
                #
                if len(os.listdir(frames_path)) == 0:
                    ts = time.time()
                    subprocess.call([ffmpeg, '-i', video_path, '-r', str(fps), '-start_number', '0', out_frame_pattern, '-c:a', 'copy', audio_dump_path])
                    print(f'Time elapsed extracting frames and audio: {(time.time() - ts):.3f} [s].')
                else:
                    print('Skip splitting video into frames and audio. Already done.')

                #
                # step2: Compute person masks and processed/refined masks
                #
                ts = time.time()
                mask_dirs = extract_person_masks_from_frames(processed_video_dir, frames_path, args.segmentation_batch_size, mask_extension)
                print(f'Time elapsed computing masks: {(time.time() - ts):.3f} [s].')

                #
                # step3: Compute stylized frames
                #
                ts = time.time()
                style_dir = stylization(frames_path)
                print(f'Time elapsed stylizing imagery: {(time.time() - ts):.3f} [s].')

                #
                # step4: Combine stylized frames and masks
                #
                relevant_directories = {}
                relevant_directories['frames_path'] = frames_path
                relevant_directories.update(mask_dirs)
                relevant_directories.update(style_dir)
                relevant_directories['audio_path'] = audio_dump_path

                ts = time.time()
                combined_dirs = stylized_frames_mask_combiner(relevant_directories, frame_extension)
                print(f'Time elapsed masking stylized imagery: {(time.time() - ts):.3f} [s].')

                #
                # step5: Create videos
                #
                relevant_directories.update(combined_dirs)

                ts = time.time()
                video_metadata = {'fps': fps}
                create_videos(video_metadata, relevant_directories, frame_name_format, args.delete_source_imagery)
                print(f'Time elapsed creating videos: {(time.time() - ts):.3f} [s].')
            else:
                raise Exception(f'{ffmpeg} not found in the system path, aborting.')
