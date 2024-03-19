from os import listdir
from os.path import isfile, join, splitext
import imageio.v2 as io
import rawpy
import cv2
import glob


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


def image_converter(original_folder, save_folder, new_format='.png', original_format=None,  title_multiplier=False,
                    raw_processing_args=None):
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


def video_maker(video_path, frames_path, fps):
    """
    Creates a video from a file of frames
    :param video_path:
    :param frames_path:
    :return:
    """
    img_array = []
    size = (0, 0)
    for filename in glob.glob(frames_path + '*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()