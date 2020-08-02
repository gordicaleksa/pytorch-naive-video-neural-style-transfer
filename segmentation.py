import argparse


from torchvision import models
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import cv2 as cv
import time
import shutil


SUPPORTED_VIDEOS = ['.mp4']


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mp4_filter(vid_name):
    return vid_name.endswith('.mp4')


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


# img_path = r"C:\tmp_data_dir\person.jpeg"
# dump_mask_single_img(dlab, path=img_path, show_orig=False)
def dump_mask_single_img(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = transforms.Compose([T.Resize(640),
                   #T.CenterCrop(224),
                   T.ToTensor(),
                   T.Normalize(mean=IMAGENET_MEAN_1,
                               std=IMAGENET_STD_1)])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  # rgb = decode_segmap(om)
  gray = om == 15
  tmp = Image.fromarray(np.uint8(gray*255))
  tmp.save(os.path.join(os.path.split(path)[0], 'mask.png'))

  # plt.imshow(gray); plt.show()
  # plt.imshow(tmp); plt.axis('off'); plt.show()


def extract_masks_from_frames(model, device, processed_video_dir, frames_path, batch_size, format):
    masks_dump_path = os.path.join(processed_video_dir, 'masks')
    processed_masks_dump_path = os.path.join(processed_video_dir, 'processed_masks')
    os.makedirs(masks_dump_path, exist_ok=True)
    os.makedirs(processed_masks_dump_path, exist_ok=True)

    HUMAN_CLASS = 15
    transform = transforms.Compose([
        transforms.Resize(540),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    dataset = datasets.ImageFolder(os.path.join(frames_path, os.path.pardir), transform=transform)
    frames_loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch_id, (img_batch, _) in enumerate(frames_loader):
            img_batch = img_batch.to(device)

            out_cpu_batch = model(img_batch)['out'].to('cpu').numpy()
            for j, out_cpu in enumerate(out_cpu_batch):
                print('creating mask for frame #{}'.format(batch_id*batch_size+j))
                mask = np.argmax(out_cpu, axis=0) == HUMAN_CLASS

                mask = np.uint8(mask * 255)
                processed_mask = process_mask(mask, batch_id*batch_size+j)

                mask = Image.fromarray(mask)
                processed_mask = Image.fromarray(processed_mask)
                # print(np.histogram(processed_mask, bins=np.arange(256)))

                mask_path = os.path.join(masks_dump_path, str(batch_id*batch_size+j).zfill(4) + '_mask' + format)
                pmask_path = os.path.join(processed_masks_dump_path, str(batch_id*batch_size+j).zfill(4) + '_p_mask' + format)

                mask.save(mask_path)
                processed_mask.save(pmask_path)

                # plt.imshow(mask)
                # plt.show()


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def process_mask(mask, it):
    # ts = time.time()
    # step1: morphological filtering
    kernel = np.ones((13, 13), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the human component
    output = cv.connectedComponentsWithStats(opening)
    num_labels = output[0]
    labels = output[1]
    h, _ = labels.shape
    discriminant_subspace = labels[:int(h/10), :]
    bkg_index = np.bincount(discriminant_subspace.flatten()).argmax()
    stats = output[2]
    areas = []
    for i in range(0, num_labels):
        areas.append(stats[i, cv.CC_STAT_AREA])
    areas = list(zip(range(len(areas)), areas))
    areas.sort(key=lambda tup: tup[1], reverse=True)
    areas = [a for a in areas if a[0] != bkg_index]  # remove background component
    human_index = areas[0][0]  # biggest component that is not background is presumably human
    processed_mask = np.uint8((labels == human_index) * 255)

    # print('processing took: {:.3f} ms'.format((time.time()-ts)*1000))

    # plt.imshow(np.hstack([labels, processed_mask]))
    # plt.show()

    return processed_mask


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    # todo: figure out why is jpg giving me bad visual quality for dummy.mp4 video
    frame_extension = '.jpg'  # .jpg is suitable to use here - smaller size and unobservable quality loss
    mask_extension = '.png'  # don't use .jpg here! bigger size + corrupts the binary property of the mask when loaded
    frame_name_format = '%06d' + frame_extension  # e.g. 000023.jpg
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
        if os.path.isfile(maybe_video_path) and os.path.splitext(maybe_video_path)[1].lower() in SUPPORTED_VIDEOS:
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
                print(f'Time it took for masks {(time.time() - ts):.3f} [s].')
            else:
                raise Exception(f'{ffmpeg} not found in the system path, aborting.')
