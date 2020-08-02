import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return np.array(img)


if __name__=="__main__":
    root = r'C:\tmp_data_dir\YouTube\Videos\NST_3_Optimization\camera_capture\intro_part_of_outro\intro_split\clip_1\processed_masks_refined' # r"C:\tmp_data_dir\YouTube\NST\Repos\examples\fast_neural_style\my_videos\NST_v2\clip_4\processed_masks_refined"
    clear_mask_path = r"C:\tmp_data_dir\YouTube\NST\Repos\examples\fast_neural_style\my_videos\NST_v2\clip_4\clear_mask9.png"

    choice = 1

    if choice == 2:
        clear_mask = load_image(clear_mask_path)[:,:,0]
    kernel = np.ones((13, 13), np.uint8)

    for cnt, img_name in enumerate(os.listdir(root)):

        if choice == 1:
            if 0 <= cnt <= 1041:
                img_path = os.path.join(root, img_name)
                img = load_image(img_path)
                print(img.shape, np.max(img))
                img[500:, 417:700] = 255
                img = Image.fromarray(img)
                img.save(img_path)
        elif choice == 2:
            if 1147 <= cnt <= 1237:
                img_path = os.path.join(root, img_name)
                img = load_image(img_path)
                # plt.imshow(clear_mask == 255)
                # plt.show()
                img[clear_mask == 255] = 0
                img = Image.fromarray(img)
                img.save(img_path)
        else:
            img_path = os.path.join(root, img_name)
            img = load_image(img_path)
            eroded = cv.morphologyEx(img, cv.MORPH_ERODE, kernel)
            # rgb = np.zeros((540, 960, 3))
            # rgb[:,:,0] = img
            # rgb[:,:,1] = eroded
            # rgb[:,:,2] = eroded
            # plt.imshow(rgb)
            # plt.show()
            eroded = Image.fromarray(eroded)
            eroded.save(img_path)