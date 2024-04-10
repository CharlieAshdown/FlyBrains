from os import listdir
from os.path import isfile, join, splitext

import imageio.v2 as io
import rawpy
import cv2
import glob
import numpy as np
import csv


def file_combiner(file_roots, new_file_location):
    """
    Takes all the images within a list of files and combines them into one.
    :param file_roots: Array of roots to the file.
    :param new_file_location: Location to store all files.
    :return:
    """
    for mul, file_root in enumerate(file_roots):
        files = [f for f in listdir(file_root) if isfile(join(file_root, f))]
        for im_num, file in enumerate(files):
            image = io.imread(file_root+file)
            io.imsave(new_file_location + str("{:04d}".format(im_num+40*mul)) + splitext(file)[1],
                      image)

    return


def image_converter(original_folder, save_folder, new_format='.png', original_format=None,
                    title_multiplier=False, raw_processing_args=None):
    """
    Mass converts images to a different type.
    :param original_folder: The path to take the images from.
    :param save_folder: The path to store the new images.
    :param new_format: Format to convert to.
    :param original_format: Format to convert from, defaults to None
    :param title_multiplier: Number to multiply the title of the image by.
           Defaults to false if not needed.
    :param raw_processing_args: Arguments from rawpy processing
    :return:
    """
    raw_formats = ['.dng', '.gpr', '.nef']
    only_files = [f for f in listdir(original_folder) if isfile(join(original_folder, f))]
    for im_num, path in enumerate(only_files):
        # For raw image processing
        if original_format in raw_formats or splitext(path)[1] in raw_formats:
            if not raw_processing_args:
                raw_processing_args = {
                    "no_auto_bright": True,
                    "output_bps": 16,
                    "demosaic_algorithm": rawpy.DemosaicAlgorithm(3),
                    "output_color": rawpy.ColorSpace(2),
                    "use_camera_wb": True}
            with rawpy.imread(original_folder + "/" + path) as raw:
                image = raw.postprocess(**raw_processing_args)
        elif original_format == splitext(path)[1] or not original_format:
            image = io.imread(path)
        else:
            return
        if not title_multiplier:
            io.imsave(save_folder + splitext(path)[0] + new_format, image)
        elif isinstance(title_multiplier, int):
            io.imsave(save_folder + str("{:04d}".format(im_num + len(only_files) * title_multiplier)) + new_format, image)


def video_converter(video_path, frames_path):
    """
    Converts a video file into a series of frames.
    :param video_path: Path to the video
    :return: The fps of the video
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(frames_path + "frame_" + "{:04d}".format(count) + ".png", image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    return fps


def video_maker(video_path, frames_path, fps, encoder="avc1"):
    """
    Creates a video from a file of frames
    :param video_path:
    :param frames_path:
    :param encoder: video encoder
    :return:
    """
    img_array = []
    size = (0, 0)
    for filename in glob.glob(frames_path + '*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*encoder), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def get_led_timings(path, fps):
    timing_files = glob.glob(f"{path}/*.txt")
    if not timing_files:
        return None, None
    LED_timing_file = None
    video_timing_file = None
    for file in timing_files:
        if "LED" in file:
            LED_timing_file = file
        if "video" in file:
            video_timing_file = file
    if not LED_timing_file or not video_timing_file:
        return None, None
    with open(LED_timing_file, 'r') as f:
        LED_timing_data = list(csv.reader(f, delimiter="\t"))[0]
        LED_timing_data = dict([i.replace(' ', '').split(':') for i in LED_timing_data])
        LED_timing_data = {k: float(v) for k, v in LED_timing_data.items()}

    with open(video_timing_file, 'r') as f:
        video_timing_data = list(csv.reader(f, delimiter="\t"))[0]
        video_timing_data = dict([i.replace(' ', '').split(':') for i in video_timing_data])
        video_timing_data = {k: float(v) for k, v in video_timing_data.items()}

    led_on = round((LED_timing_data["LEDsON"] - video_timing_data["VideoON"])*fps)
    led_off = round((LED_timing_data["LEDsOFF"] - video_timing_data["VideoON"])*fps)
    return led_on, led_off


class Queue:
    """
    Implements a FIFO queue with a maximum size with numpy functionality
    """
    def __init__(self, max_size=100):
        """
        Initialise the queue
        :param max_size: Sets the maximum size of the array
        """
        self.list = []
        self.array = np.array([])
        self.max_size = max_size

    def is_full(self):
        """
        Determines whether the array is full
        :return: True if full, False if not
        """
        if len(self.array) == self.max_size:
            return True
        else:
            return False

    def put(self, value):
        """
        Puts values into array.
        :param value: Must be list
        :return:
        """
        if len(self.list) == self.max_size:
            del self.list[0]
        self.list.append(value)
        self.array = np.array(self.list)

    def mean(self):
        """
        Calculates the mean
        :return: Mean value for each portion of the list
        """
        return list(np.mean(self.array, axis=0))

    def speed(self):
        """
        Calculates the speed over the array
        :return: Speed value for each portion of the list
        """
        min = np.nanmin(self.array, axis=0)
        max = np.nanmax(self.array, axis=0)
        if len(max.shape) > 1:
            speed = np.linalg.norm((np.subtract(max, min) / self.max_size), axis=1)
        else:
            speed = (np.subtract(max, min) / self.max_size)
        return list(speed)


class new_set(set):
    """
    Adding the add and subtract functionality to sets.
    """
    def __add__(self, other):
        if type(other) is int or type(other) is float:
            return new_set(x + other for x in self)
        elif type(other) is set or tuple:
            return new_set(x + y for x, y in zip(self, other))

    def __sub__(self, other):
        if type(other) is int or type(other) is float:
            return new_set(x - other for x in self)
        elif type(other) is set or tuple:
            return new_set(x - y for x, y in zip(self, other))


def automatic_brightness_and_contrast(image, alpha=None, beta=None, clip_hist_percent=1):
    """
    Automatic brightness and contrast optimization with optional histogram clipping
    :param image:
    :param alpha:
    :param beta:
    :param clip_hist_percent:
    :return:
    """
    if not alpha:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta
