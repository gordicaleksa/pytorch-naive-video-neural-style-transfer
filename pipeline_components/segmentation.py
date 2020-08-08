import os


from torchvision import models
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv


from .constants import *


# Using functions from utils package from pytorch-nst-feedforward submodule like load_image
from utils import utils


def post_process_mask(mask):
    """
    Helper function for automatic mask (produced by the segmentation model) cleaning using heuristics.
    """

    # step1: morphological filtering (helps splitting parts that don't belong to the person blob)
    kernel = np.ones((13, 13), np.uint8)  # hardcoded 13 simply gave nice results
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the person component (biggest component after background)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)

    if num_labels > 1:
        # step2.1: find the background component
        h, _ = labels.shape  # get mask height
        # find the most common index in the upper 10% of the image - I consider that to be the background index (heuristic)
        discriminant_subspace = labels[:int(h/10), :]
        bkg_index = np.argmax(np.bincount(discriminant_subspace.flatten()))

        # step2.2: biggest component after background is person (that's a highly probable hypothesis)
        blob_areas = []
        for i in range(0, num_labels):
            blob_areas.append(stats[i, cv.CC_STAT_AREA])
        blob_areas = list(zip(range(len(blob_areas)), blob_areas))
        blob_areas.sort(key=lambda tup: tup[1], reverse=True)  # sort from biggest to smallest area components
        blob_areas = [a for a in blob_areas if a[0] != bkg_index]  # remove background component
        person_index = blob_areas[0][0]  # biggest component that is not background is presumably person
        processed_mask = np.uint8((labels == person_index) * 255)

        return processed_mask
    else:  # only 1 component found (probably background) we don't need further processing
        return opened_mask


def extract_person_masks_from_frames(processed_video_dir, frames_path, batch_size, segmentation_mask_width, mask_extension):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Currently the best segmentation model in PyTorch (officially implemented)
    segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()
    print(f'Number of trainable weights in the segmentation model: {utils.count_parameters(segmentation_model)}')

    masks_dump_path = os.path.join(processed_video_dir, 'masks')
    processed_masks_dump_path = os.path.join(processed_video_dir, 'processed_masks')
    os.makedirs(masks_dump_path, exist_ok=True)
    os.makedirs(processed_masks_dump_path, exist_ok=True)

    h, w = utils.load_image(os.path.join(frames_path, os.listdir(frames_path)[0])).shape[:2]
    segmentation_mask_height = int(h * (segmentation_mask_width / w))
    transform = transforms.Compose([
        transforms.Resize((segmentation_mask_height, segmentation_mask_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    dataset = datasets.ImageFolder(os.path.join(frames_path, os.path.pardir), transform=transform)
    frames_loader = DataLoader(dataset, batch_size=batch_size)

    if len(os.listdir(masks_dump_path)) == 0 and len(os.listdir(processed_masks_dump_path)) == 0:
        print('*' * 20, 'Person segmentation stage started', '*' * 20)
        with torch.no_grad():
            try:
                processed_imgs_cnt = 0
                for batch_id, (img_batch, _) in enumerate(frames_loader):
                    processed_imgs_cnt += len(img_batch)
                    print(f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(dataset)} processed images).')
                    img_batch = img_batch.to(device)  # shape: (N, 3, H, W)
                    result_batch = segmentation_model(img_batch)['out'].to('cpu').numpy()  # shape: (N, 21, H, W) (21 - PASCAL VOC classes)
                    for j, out_cpu in enumerate(result_batch):
                        # When for the pixel position (x, y) the biggest (un-normalized) probability
                        # lies in the channel PERSON_CHANNEL_INDEX we set the mask pixel to True
                        mask = np.argmax(out_cpu, axis=0) == PERSON_CHANNEL_INDEX
                        mask = np.uint8(mask * 255)  # convert from bool to [0, 255] black & white image

                        processed_mask = post_process_mask(mask)  # simple heuristics (connected components, etc.)

                        filename = str(batch_id*batch_size+j).zfill(FILE_NAME_NUM_DIGITS) + mask_extension
                        cv.imwrite(os.path.join(masks_dump_path, filename), mask)
                        cv.imwrite(os.path.join(processed_masks_dump_path, filename), processed_mask)
            except Exception as e:
                print(str(e))
                print(f'Try using smaller segmentation batch size than the current one ({batch_size} images in batch).')
                exit(ERROR_CODE)
    else:
        print('Skipping mask computation, already done.')

    return {'processed_masks_dump_path': processed_masks_dump_path}