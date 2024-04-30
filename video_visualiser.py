import os
import glob
import shutil
import sys

import cv2
import torch
import numpy as np
from utilities import video_converter, get_led_timings, new_set, automatic_brightness_and_contrast, video_maker
from os.path import splitext
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import Pad, CenterCrop
from torchvision.transforms.functional import to_pil_image, pil_to_tensor


def video_visualiser(path=None, brighten=True, black_and_white=False, save_images=False):
    """
    Increases brightness of video and add border when optogenetic LEDs on
    :param path: Path to video
    :param brighten: Whether to brighten the image
    :param black_and_white: Whether to convert image to black and white
    :return: Path to formatted video or None if failed
    """
    if not path:
        try:
            parent_dir = sys.argv[1]

        except IndexError:
            parent_dir = input("Input directory: ")

    elif os.path.isdir(path):
        parent_dir = path
    else: return None
    save_dir = parent_dir
    pad_amount = 10
    video_name = glob.glob(f"{parent_dir}/*.h264")[0]

    # Path
    frames_path = os.path.join(parent_dir, "frames")
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    output_path = os.path.join(parent_dir, "bounded_frames")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    fps = video_converter(os.path.join(parent_dir, video_name), frames_path + "/")
    led_on, led_off = get_led_timings(parent_dir, fps)
    alpha, beta = None, None

    image_paths = list(sorted(os.listdir(frames_path)))
    for image_num, image_path in enumerate(image_paths):
        image = read_image(frames_path + "/" + image_path)

        if black_and_white:
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            r, g, b = cv2.split(image)
            image = np.dstack((r, r, r))
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        if brighten:
            image = image.numpy()
            image = np.transpose(image, (1, 2, 0))
            image, alpha, beta = automatic_brightness_and_contrast(image, alpha=alpha, beta=beta)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        if led_on is not None and led_on <= image_num <= led_off:
            image = to_pil_image(image)
            crop = CenterCrop((new_set(image.size) - 20))
            image = crop(image)
            pad = Pad(pad_amount, fill=(255, 0, 0), padding_mode="constant")
            image = pad(image)
            image = pil_to_tensor(image)

        save_image(image.float(), os.path.join(output_path + "/", image_path), normalize=True)

    video_maker(os.path.join(save_dir, splitext(video_name)[0] + "_brightened" + ".mp4"),
                output_path + "/", fps, encoder="mp4v")
    if not save_images:
        shutil.rmtree(frames_path)
        shutil.rmtree(output_path)
    if black_and_white:
        return os.path.join(save_dir, splitext(video_name)[0] + "_single_channel" + ".mp4")
    elif brighten:
        return os.path.join(save_dir, splitext(video_name)[0] + "_brightened" + ".mp4")


if __name__ == "__main__":

    video_visualiser(path="D:/Flybrains/samples/samples_25_03_2024/test_05/",
                     brighten=True,
                     black_and_white=True,
                     save_images=True)
