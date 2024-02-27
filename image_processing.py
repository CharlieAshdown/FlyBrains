import numpy as np
import rawpy
import imageio
import cv2
import time
from os import listdir
from os.path import isfile, join
from sklearn.cluster import DBSCAN


def separate_clusters(coords, thresh):
    db = DBSCAN(eps=10, min_samples=3).fit(coords)
    labels = db.labels_

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    results = {}

    for k in unique_labels:
        if k != -1:
            class_member_mask = labels == k
            result = np.zeros(thresh.shape)
            class_coords = coords[class_member_mask & core_samples_mask]
            for i in range(len(class_coords)):
                result[class_coords[i, 0], class_coords[i, 1]] = 1
            results["DBSCAN mask " + str(k)] = {}
            results["DBSCAN mask " + str(k)]["image"] = result
            results["DBSCAN mask " + str(k)]["coords"] = class_coords
            results["DBSCAN mask " + str(k)]["area"] = len(class_coords)
    # print(areas)
    return results


def find_larvae(r, g, b):

    _, thresh_b_flies = cv2.threshold(b, 28000, 65535, cv2.THRESH_BINARY)
    _, thresh_b_background = cv2.threshold(b, 50000, 65535, cv2.THRESH_BINARY)
    size = b.size
    # _, thresh_g = cv2.threshold(g, 30000, 65535, cv2.THRESH_BINARY)
    # _, thresh_r = cv2.threshold(r, 20000, 65535, cv2.THRESH_BINARY)

    """
    imS = cv2.resize(thresh_b_flies, (812, 608))
    cv2.imshow("Threshold Blue Flies Image", imS)
    imS = cv2.resize(thresh_b_background, (812, 608))
    cv2.imshow("Threshold Blue Background Image", imS)
    imS = cv2.resize(thresh_g, (812, 608))
    cv2.imshow("Threshold Green Image", imS)
    imS = cv2.resize(thresh_r, (812, 608))
    cv2.imshow("Threshold Red Image", imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    thresh_b_flies_coords = np.argwhere(thresh_b_flies == 65535)
    thresh_b_background_coords = np.argwhere(thresh_b_background == 65535)

    """ 
        DBSCAN algorithm
    """

    clusters = {"blue": separate_clusters(thresh_b_flies_coords, thresh_b_flies)}

    all_background_coords_array = thresh_b_background_coords.tolist()

    larvae = {}
    for t, frame in enumerate(clusters["blue"]):
        blue_frame_coords = clusters["blue"][frame]["coords"].tolist()
        inter = any(i in blue_frame_coords for i in all_background_coords_array)
        # test = all_green_coords[inter] # gives coordinates if inter changed to list
        if clusters["blue"][frame]["area"] > 2000 and not inter:
            larvae["frame" + str(t)] = {}
            larvae["frame" + str(t)]["image"] = clusters["blue"][frame]["image"]
            larvae["frame" + str(t)]["area"] = clusters["blue"][frame]["area"]
            larvae["frame" + str(t)]["coords"] = clusters["blue"][frame]["coords"]
            # imS = cv2.resize(larvae["frame" + str(t)]["image"], (812, 608))
            # cv2.imshow("frame" + str(t), imS)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return larvae


def crop_flies(images):
    start = time.time()
    for sample in images:
        r, g, b = cv2.split(images[sample])
        found_larvae[sample] = find_larvae(r, g, b)
        for n, larvae in enumerate(found_larvae[sample]):
            found_larvae[sample][larvae]["crop"] = {}
            maximums = np.max(found_larvae[sample][larvae]["coords"], axis=0)
            minimums = np.min(found_larvae[sample][larvae]["coords"], axis=0)
            found_larvae[sample][larvae]["crop"]["x_max"] = maximums[1] + border
            found_larvae[sample][larvae]["crop"]["x_min"] = minimums[1] - border
            found_larvae[sample][larvae]["crop"]["y_max"] = maximums[0] + border
            found_larvae[sample][larvae]["crop"]["y_min"] = minimums[0] - border

            crop_sample = images[sample][found_larvae[sample][larvae]["crop"]["y_min"]:
                                         found_larvae[sample][larvae]["crop"]["y_max"],
                                         found_larvae[sample][larvae]["crop"]["x_min"]:
                                         found_larvae[sample][larvae]["crop"]["x_max"]]

            imageio.imsave(save_path + sample + "_" + str(n) + '.tiff', crop_sample)

    end = time.time()

    print("Time taken: " + str(end - start))


def make_mask(images):
    colours = {"red": [0, 0, 1],
               "green": [0, 1, 0],
               "blue": [1, 0, 0],
               "turquoise": [1, 1, 0],
               "magenta": [1, 0, 1],
               "yellow": [0, 1, 1]}
    start = time.time()
    for sample in images:
        r, g, b = cv2.split(images[sample])
        b_mask = np.zeros(b.shape)
        g_mask = np.zeros(g.shape)
        r_mask = np.zeros(r.shape)
        found_larvae[sample] = find_larvae(r, g, b)
        for (_, larvae), (_, colour) in zip(found_larvae[sample].items(), colours.items()):
            b_mask += (larvae["image"] * colour[0])*255
            g_mask += (larvae["image"] * colour[1])*255
            r_mask += (larvae["image"] * colour[2])*255
        image_merge = np.uint8(cv2.merge([r_mask, g_mask, b_mask]))
        imageio.imsave(save_path + sample + '_mask.png', image_merge)
        """
        imS = cv2.resize(image_merge, (812, 608))
        cv2.imshow("mask", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    end = time.time()

    print("Time taken: " + str(end - start))


mypath = "C:/Users/Charlie/Documents/samples/samples_15_02_2024/test_14"
save_path = "C:/Users/Charlie/Documents/samples/samples_15_02_2024/test_14_masks/"

only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

samples = {}
found_larvae = {}
border = 20

for path in only_files:
    with rawpy.imread(mypath + "/" + path) as raw:
        rgb = raw.postprocess(no_auto_bright=True, output_bps=16,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm(3),
                              output_color=rawpy.ColorSpace(2), use_camera_wb=True)
        samples[path] = rgb

make_mask(samples)