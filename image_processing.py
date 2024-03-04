import numpy as np
import rawpy
import imageio
import cv2
import time
import scipy
from os import listdir
from os.path import isfile, join, splitext
from sklearn.cluster import DBSCAN


class ImageProcessing:
    """
    Performs the image processing necessary to use the sample data in an AI.
    """
    def __init__(self, samples_path, training_images_path, epsilon=10, min_samples=3, save_images=False):
        """
        Opens the sample images from the folder and stores them in the samples dict

        :param samples_path: The path to the sample images folder
        :param training_images_path: The path to save the images for training
        :param epsilon: The max distance between nodes for DBSCAN
        :param min_samples: The minimum number of samples to determine a class with DBSCAN
        :param save_images: Whether to save the images as PNGs or not.
        """
        only_files = [f for f in listdir(samples_path) if isfile(join(samples_path, f))]
        self.samples = {}
        self.found_larvae = {}
        self.epsilon = epsilon
        self.min_samples = min_samples
        for path in only_files:
            with rawpy.imread(samples_path + "/" + path) as raw:
                rgb = raw.postprocess(no_auto_bright=True, output_bps=16,
                                      demosaic_algorithm=rawpy.DemosaicAlgorithm(3),
                                      output_color=rawpy.ColorSpace(2), use_camera_wb=True)
                self.samples[splitext(path)[0]] = rgb
                if save_images:
                    imageio.imsave(training_images_path + "/" + splitext(path)[0] + '.png',
                                   ((rgb / rgb.max())*255).astype(np.uint8))

    def set_dbscan(self, new_epsilon=10, new_min_samples=3):
        """
        Sets the values of epsilon and min_samples
        :param new_epsilon: New value of epsilon
        :param new_min_samples: New value of min_samples
        :return:
        """
        self.epsilon = new_epsilon
        self.min_samples = new_min_samples

    def _separate_clusters(self, coords, size):
        """
        Performs the DBSCAN algorithm and returns the cluster details

        :param coords: The thresholded image as an array of coordinates.
        :return: A dictionary of the image, coordinates and area of the clusters found.
        """
        start_db = time.time()
        db = DBSCAN(eps=self.epsilon, min_samples=self.min_samples).fit(coords)
        end_db = time.time()
        print("DB time: " + str(end_db-start_db))
        labels = db.labels_

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        results = {}

        for k in unique_labels:
            if k != -1:
                class_member_mask = labels == k
                result = np.zeros(size)
                class_coords = coords[class_member_mask & core_samples_mask]
                for i in range(len(class_coords)):
                    result[class_coords[i, 0], class_coords[i, 1]] = 1
                results["DBSCAN mask " + str(k)] = {}
                results["DBSCAN mask " + str(k)]["image"] = result
                results["DBSCAN mask " + str(k)]["coords"] = class_coords
                results["DBSCAN mask " + str(k)]["area"] = len(class_coords)
        return results

    def find_background(self):
        r, g, b = cv2.split(self.samples["000"])
        background = cv2.GaussianBlur(b, (1501, 1201), 71)
        _, thresh_background = cv2.threshold(background, 14000, 65535, cv2.THRESH_BINARY)
        thresh_background = scipy.signal.medfilt2d(thresh_background, 7)
        return thresh_background

    def find_larvae(self, b, background):
        """
        Finds the larvae in the image and returns the image, area and coordinates which
        represent the larvae.

        :param b: The blue channel from the image
        :param background: The background of the image
        :return: A dict containing the image area and coordinates of the larvae
        """
        # b_background = cv2.GaussianBlur(b, (1501, 1201), 71)
        _, thresh_b_flies = cv2.threshold(b, 17000, 65535, cv2.THRESH_BINARY)
        # _, thresh_b_background = cv2.threshold(b_background, 13000, 65535, cv2.THRESH_BINARY)

        thresh_b_flies = scipy.signal.medfilt2d(thresh_b_flies, 5)
        # thresh_b_background = scipy.signal.medfilt2d(thresh_b_background, 7)

        """
        imS = cv2.resize(thresh_b_flies, (812, 608))
        cv2.imshow("Threshold Blue Flies Image", imS)
        imS = cv2.resize(background, (812, 608))
        cv2.imshow("Threshold Blue Background Image", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        thresh_b_flies_coords = np.argwhere(thresh_b_flies == 65535)
        # thresh_b_background_coords = np.argwhere(thresh_b_background == 65535)

        clusters = {"blue": self._separate_clusters(thresh_b_flies_coords, b.shape)}

        # all_background_coords_array = thresh_b_background_coords.tolist()

        larvae = {}
        start_larvae = time.time()

        inter_time = 0
        if_time = 0

        for t, frame in enumerate(clusters["blue"]):
            # blue_frame_coords = clusters["blue"][frame]["coords"].tolist()
            start_inter = time.time()

            # inter = any(i in blue_frame_coords for i in all_background_coords_array)

            inter = np.any(np.logical_and(clusters["blue"][frame]["image"].astype(np.bool_),
                                          background.astype(np.bool_)))

            end_inter = time.time()
            inter_time += end_inter - start_inter

            start_if = time.time()
            if clusters["blue"][frame]["area"] > 1500 and not inter:
                larvae["frame" + str(t)] = {}
                larvae["frame" + str(t)]["image"] = clusters["blue"][frame]["image"]
                larvae["frame" + str(t)]["area"] = clusters["blue"][frame]["area"]
                larvae["frame" + str(t)]["coords"] = clusters["blue"][frame]["coords"]
            end_if = time.time()
            if_time += (end_if - start_if)
        end_larvae = time.time()
        print("Inter time: " + str(inter_time))
        print("If time: " + str(if_time))
        print("Larvae finding time: " + str(end_larvae-start_larvae))
        return larvae

    def crop_flies(self, border, save_path):
        """
        Finds the larvae in the sample image and saves individual images
        of the cropped larvae.

        :param border: The border around the cropped image
        :param save_path: The path to save the cropped image
        :return:
        """
        start = time.time()
        for sample in self.samples:
            r, g, b = cv2.split(self.samples[sample])
            self.found_larvae[sample] = self.find_larvae(b)
            for n, larvae in enumerate(self.found_larvae[sample]):
                self.found_larvae[sample][larvae]["crop"] = {}
                maximums = np.max(self.found_larvae[sample][larvae]["coords"], axis=0)
                minimums = np.min(self.found_larvae[sample][larvae]["coords"], axis=0)
                self.found_larvae[sample][larvae]["crop"]["x_max"] = maximums[1] + border
                self.found_larvae[sample][larvae]["crop"]["x_min"] = minimums[1] - border
                self.found_larvae[sample][larvae]["crop"]["y_max"] = maximums[0] + border
                self.found_larvae[sample][larvae]["crop"]["y_min"] = minimums[0] - border

                crop_sample = self.samples[sample][self.found_larvae[sample][larvae]["crop"]["y_min"]:
                                                   self.found_larvae[sample][larvae]["crop"]["y_max"],
                                                   self.found_larvae[sample][larvae]["crop"]["x_min"]:
                                                   self.found_larvae[sample][larvae]["crop"]["x_max"]]

                imageio.imsave(save_path + sample + "_" + str(n) + '.tiff', crop_sample)

        end = time.time()

        print("Time taken: " + str(end - start))

    def make_mask(self, save_path):
        """
        Finds the individual larvae in the sample image and saves a mask of the larvae
        each represented by a different colour.

        :param save_path: The path to save the images.
        :return:
        """
        colours = {"red": [0, 0, 1],
                   "green": [0, 1, 0],
                   "blue": [1, 0, 0],
                   "turquoise": [1, 1, 0],
                   "magenta": [1, 0, 1],
                   "yellow": [0, 1, 1]}
        start = time.time()
        background = self.find_background()
        for sample in self.samples:
            r, g, b = cv2.split(self.samples[sample])
            b_mask = np.zeros(b.shape)
            g_mask = np.zeros(g.shape)
            r_mask = np.zeros(r.shape)
            self.found_larvae[sample] = self.find_larvae(b, background)
            for (_, larvae), (_, colour) in zip(self.found_larvae[sample].items(), colours.items()):
                b_mask += (larvae["image"] * colour[0]) * 255
                g_mask += (larvae["image"] * colour[1]) * 255
                r_mask += (larvae["image"] * colour[2]) * 255
            # back = np.where(b_mask == 0) and np.where(g_mask == 0) and np.where(r_mask == 0)
            image_merge = np.uint8(cv2.merge([r_mask, g_mask, b_mask]))
            imageio.imsave(save_path + "/" + sample + '_mask.png', image_merge)

        end = time.time()

        print("Time taken: " + str(end - start))


if __name__ == "__main__":
    start_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/test_1"
    training_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_1_training/images"
    mask_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_1_training/masks"
    image_processing = ImageProcessing(start_path, training_path, save_images=False)
    image_processing.make_mask(mask_path)
