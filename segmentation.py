import argparse
import subprocess
import time
import os
import shutil


from torchvision import models
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


SUPPORTED_VIDEO_EXTENSIONS = ['.mp4']


FILE_NAME_NUM_DIGITS = 6  # number of digits in the frame/mask names, e.g. for 6: '000023.jpg'


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def post_process_mask(mask):
    # step1: morphological filtering (helps splitting parts that don't belong to the person blob)
    kernel = np.ones((13, 13), np.uint8)  # hardcoded 13 simply gave nice results
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the person component (biggest component after background)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)

    # step2.1: find the background component
    h, _ = labels.shape  # get mask height
    # find the most common index in the upper 10% of the image - I consider that to be the background index (heuristic)
    discriminant_subspace = labels[:int(h/10), :]
    bkg_index = np.argmax(np.bincount(discriminant_subspace.flatten()))

    # step2.2: biggest component after background is person (hypothesis)
    blob_areas = []
    for i in range(0, num_labels):
        blob_areas.append(stats[i, cv.CC_STAT_AREA])
    blob_areas = list(zip(range(len(blob_areas)), blob_areas))
    blob_areas.sort(key=lambda tup: tup[1], reverse=True)  # sort from biggest to smallest area components
    blob_areas = [a for a in blob_areas if a[0] != bkg_index]  # remove background component
    person_index = blob_areas[0][0]  # biggest component that is not background is presumably person
    processed_mask = np.uint8((labels == person_index) * 255)

    return processed_mask


def extract_masks_from_frames(model, device, processed_video_dir, frames_path, batch_size, mask_extension):
    masks_dump_path = os.path.join(processed_video_dir, 'masks')
    processed_masks_dump_path = os.path.join(processed_video_dir, 'processed_masks')
    os.makedirs(masks_dump_path, exist_ok=True)
    os.makedirs(processed_masks_dump_path, exist_ok=True)

    PERSON_CHANNEL_INDEX = 15
    transform = transforms.Compose([
        transforms.Resize(540),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    dataset = datasets.ImageFolder(os.path.join(frames_path, os.path.pardir), transform=transform)
    frames_loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch_id, (img_batch, _) in enumerate(frames_loader):
            print(f'Processing batch {batch_id + 1}.')
            img_batch = img_batch.to(device)  # shape: (N, 3, H, W)
            result_batch = model(img_batch)['out'].to('cpu').numpy()  # shape: (N, 21, H, W) (21 - PASCAL VOC classes)
            for j, out_cpu in enumerate(result_batch):
                # When for the pixel position (x, y) the biggest (un-normalized) probability
                # lies in the channel PERSON_CHANNEL_INDEX we set the mask pixel to True
                mask = np.argmax(out_cpu, axis=0) == PERSON_CHANNEL_INDEX
                mask = np.uint8(mask * 255)  # convert from bool to [0, 255] black & white image

                processed_mask = post_process_mask(mask)  # simple heuristics (connected components, etc.)

                filename = str(batch_id*batch_size+j).zfill(FILE_NAME_NUM_DIGITS) + mask_extension
                # ::-1 because opencv expects BGR (and not RGB) format...
                cv.imwrite(os.path.join(masks_dump_path, filename), mask)
                cv.imwrite(os.path.join(processed_masks_dump_path, filename), processed_mask)


# todo: should I add fast NST as a submodule project?
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Currently the best segmentation model in PyTorch
    segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()

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

                if len(os.listdir(frames_path)) == 0:
                    subprocess.call([ffmpeg, '-i', video_path, '-r', str(fps), out_frame_pattern, '-c:a', 'copy', audio_dump_path])
                    print('Done splitting video into frames and extracting audio file.')
                else:
                    print('Skip splitting video into frames and audio. Already done.')

                ts = time.time()
                extract_masks_from_frames(segmentation_model, device, processed_video_dir, frames_path, args.segmentation_batch_size, mask_extension)
                print(f'Time it took for computing masks {(time.time() - ts):.3f} [s].')
            else:
                raise Exception(f'{ffmpeg} not found in the system path, aborting.')
