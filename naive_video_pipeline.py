import argparse
import subprocess
import time
import os
import shutil
import sys
# Enables this project to see packages from pytorch-nst-feedforward submodule (e.g. utils)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch-nst-feedforward'))


import cv2 as cv


from pipeline_components.segmentation import extract_person_masks_from_frames
from pipeline_components.video_creation import create_videos
from pipeline_components.constants import *
from pipeline_components.nst_stylization import stylization
from pipeline_components.compositor import stylized_frames_mask_combiner


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
                    subprocess.call([ffmpeg, '-i', video_path, '-r', str(fps), '-start_number', '0', out_frame_pattern, '-c:a', 'copy', audio_dump_path])
                else:
                    print('Skip splitting video into frames and audio, already done.')
                print('Stage 1/5 done (split video into frames and audio file).')

                #
                # step2: Compute person masks and processed/refined masks
                #
                ts = time.time()
                mask_dirs = extract_person_masks_from_frames(processed_video_dir, frames_path, args.segmentation_batch_size, mask_extension)
                print('Stage 2/5 done (create person segmentation masks.')
                print(f'Time elapsed computing masks: {(time.time() - ts):.3f} [s].')

                #
                # step3: Compute stylized frames
                #
                ts = time.time()
                style_dir = stylization(frames_path)
                print('Stage 3/5 done (stylize dumped video frames).')
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
                print('Stage 4/5 done (combine masks with stylized frames).')
                print(f'Time elapsed masking stylized imagery: {(time.time() - ts):.3f} [s].')

                #
                # step5: Create videos
                #
                relevant_directories.update(combined_dirs)

                video_metadata = {'fps': fps}
                create_videos(video_metadata, relevant_directories, frame_name_format, args.delete_source_imagery)
                print('Stage 5/5 done (create videos from overlayed stylized frames).')
            else:
                raise Exception(f'{ffmpeg} not found in the system path, aborting.')
