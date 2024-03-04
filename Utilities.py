from os import listdir
from os.path import isfile, join, splitext
import imageio.v2 as io


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


if __name__ == "__main__":

    images_roots = ["C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_1_training/masks/",
                    "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_2_training/masks/",
                    "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_3_training/masks/"]

    all_images = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/all_training/masks/"

    file_combiner(images_roots, all_images)
