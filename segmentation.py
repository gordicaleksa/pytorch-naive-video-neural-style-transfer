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


def is_big_resolution(content_pics):
    img_path = os.path.join(content_pics, os.listdir(content_pics)[0])
    img = utils.load_image(img_path)
    h, w = img.shape[:2]
    return w == 1920


def modify_paths(paths):
    new_paths = []
    for path in paths:
        base, name = os.path.split(path)
        name = '_res_' + name
        new_path = os.path.join(base, name)
        new_paths.append(new_path)

    return new_paths


#
# stylization part
#

# device = torch.device("cuda" if args.cuda else "cpu")
#
# content_transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Lambda(lambda x: x.mul(255))
# ])
# dataset = datasets.ImageFolder(frames_path, transform=content_transform)
# num_of_frames = len(dataset)
# using_big_res = is_big_resolution(content_pics)
# batch_size = 3 if using_big_res else 14
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# print('Using batch size = {}'.format(batch_size))
#
# dump_dest = os.path.join(dump_path, 'stylized')
# os.makedirs(dump_dest, exist_ok=True)
#
# if args.model.endswith(".onnx"):
#     # output = stylize_onnx_caffe2(content_image, args)
#     print('nicee.')
# else:
#     with torch.no_grad():
#         style_model = TransformerNet()
#         state_dict = torch.load(args.model)
#         # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
#         for k in list(state_dict.keys()):
#             if re.search(r'in\d+\.running_(mean|var)$', k):
#                 del state_dict[k]
#         style_model.load_state_dict(state_dict)
#         style_model.to(device)
#         if args.export_onnx:
#             assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
#             # output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
#         else:
#             if len(os.listdir(dump_dest)) == 0:
#                 for i, (imgs, labels) in enumerate(loader):
#                     imgs = imgs.to(device)
#                     out_cpu_batch = style_model(imgs).cpu().numpy()
#                     ts = time.time()
#                     for j, styled_img in enumerate(out_cpu_batch):
#                         out_path = os.path.join(dump_dest, str(i*batch_size+j).zfill(4) + format)
#                         utils.save_image_from_vid(out_path, styled_img)
#                     print('{:04d}/{:04d} , processing batch took {:.3f}'.format((i+1)*batch_size, num_of_frames, time.time()-ts))
#             else:
#                 print('Skipping, already stilyzed.')

#
# end of stylization part
#


def stylization():
    print('dummy yet to implement')
    return {'dummy': -1}


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
        for cnt, (name1, name2, name3) in enumerate(zip(os.listdir(stylized_frames_dir), os.listdir(mask_frames_dir), os.listdir(overlay_frames_dir))):
            s_img_path = os.path.join(stylized_frames_dir, name1)  # stylized original frame image
            m_img_path = os.path.join(mask_frames_dir, name2)  # mask image
            o_img_path = os.path.join(overlay_frames_dir, name3)  # overlay image

            # todo: we'll import load_image from the submodule if that path turns out feasible
            # load input imagery
            s_img = utils.load_image(s_img_path)
            m_img = utils.load_image(m_img_path)
            s_h, s_w = s_img.shape[:2]
            m_img = cv.resize(m_img, (s_w, s_h))
            o_img = utils.load_image(o_img_path)

            # prepare canvas imagery
            combined_img_background = s_img.copy()
            combined_img_person = s_img.copy()

            # create masks
            background_mask = m_img == 0
            person_mask = m_img == 255

            # apply masks
            combined_img_background[background_mask] = o_img[background_mask]
            combined_img_person[person_mask] = o_img[person_mask]

            # save combined imagery
            combined_img_background_path = os.path.join(dump_path_bkg_masked, str(cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            combined_img_person_path = os.path.join(dump_path_person_masked, str(cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            cv.imwrite(combined_img_background_path, combined_img_background)
            cv.imwrite(combined_img_person_path, combined_img_person)
    else:
        print('Skipping combining with masks, already done.')


def create_videos():
    pic_path_pattern_bkg = os.path.join(final_dest_bkg, 'combined_%04d' + format)
    pic_path_pattern_bkg_inv = os.path.join(final_dest_bkg_inv, 'combined_%04d' + format)
    pic_path_pattern_stylized = os.path.join(dump_dest, '%04d' + format)
    out_video_path_bkg = os.path.join(final_dest_bkg, video_name + '_stylized_human_' + model_name + '.mp4')
    out_video_path_bkg_inv = os.path.join(final_dest_bkg_inv, video_name + '_stylized_bkg_' + model_name + '.mp4')
    out_video_path_stylized = os.path.join(dump_dest, video_name + '_full-style_' + model_name + '.mp4')

    # after image2 it went like this: '-s', '1920x1080' '960x540'
    subprocess.call(
        [ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_bkg, '-i', in_audio_path, '-vcodec',
         'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_bkg])
    subprocess.call(
        [ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_bkg_inv, '-i', in_audio_path, '-vcodec',
         'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_bkg_inv])
    subprocess.call(
        [ffmpeg_path, '-r', str(30), '-f', 'image2', '-i', pic_path_pattern_stylized, '-i', in_audio_path, '-vcodec',
         'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out_video_path_stylized])
    print('Creating videos done.')

    if not using_big_res:
        resized_videos_paths = modify_paths([out_video_path_bkg, out_video_path_bkg_inv, out_video_path_stylized])
        subprocess.call([ffmpeg_path, '-i', out_video_path_bkg, '-vf', 'scale=1920:1080', resized_videos_paths[0]])
        subprocess.call([ffmpeg_path, '-i', out_video_path_bkg_inv, '-vf', 'scale=1920:1080', resized_videos_paths[1]])
        subprocess.call([ffmpeg_path, '-i', out_video_path_stylized, '-vf', 'scale=1920:1080', resized_videos_paths[2]])
        print('Done resizing videos')

    if args.should_delete_images:
        [os.remove(os.path.join(final_dest_bkg, file)) for file in os.listdir(final_dest_bkg) if file.endswith(format)]
        [os.remove(os.path.join(final_dest_bkg_inv, file)) for file in os.listdir(final_dest_bkg_inv) if
         file.endswith(format)]
        # todo: tmp disabled
        # [os.remove(os.path.join(dump_dest, file)) for file in os.listdir(dump_dest) if file.endswith(format)]
        print('Deleting images done.')
    else:
        print('Won"t delete images')


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

    if len(os.listdir(masks_dump_path)) == 0 and len(os.listdir(processed_masks_dump_path)) == 0:
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
                    cv.imwrite(os.path.join(masks_dump_path, filename), mask)
                    cv.imwrite(os.path.join(processed_masks_dump_path, filename), processed_mask)
    else:
        print('Skipping mask computation, already done.')

    return {'processed_masks_dump_path': processed_masks_dump_path}


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
                    subprocess.call([ffmpeg, '-i', video_path, '-r', str(fps), out_frame_pattern, '-c:a', 'copy', audio_dump_path])
                    print(f'Time elapsed extracting frames and audio: {(time.time() - ts):.3f} [s].')
                else:
                    print('Skip splitting video into frames and audio. Already done.')

                #
                # step2: Compute person masks and processed/refined masks
                #
                ts = time.time()
                mask_dirs = extract_masks_from_frames(segmentation_model, device, processed_video_dir, frames_path, args.segmentation_batch_size, mask_extension)
                print(f'Time elapsed computing masks: {(time.time() - ts):.3f} [s].')

                #
                # step3: Compute stylized frames
                #
                ts = time.time()
                style_dir = stylization()
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
                stylized_frames_mask_combiner(relevant_directories, frame_extension)
                print(f'Time elapsed masking stylized imagery: {(time.time() - ts):.3f} [s].')
                #
                # step5: Create videos
                #
            else:
                raise Exception(f'{ffmpeg} not found in the system path, aborting.')
