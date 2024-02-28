import cv2

from image_processing import ImageProcessing

start_path = "C:/Users/Charlie/Documents/samples/samples_15_02_2024/dbscan_test"
end_path = start_path

processing = ImageProcessing(samples_path=start_path, epsilon=10, min_samples=5)
processing.make_mask(end_path)

"""
processing.set_dbscan(new_epsilon=5)
processing.make_mask(end_path)

processing.set_dbscan(new_epsilon=5)
processing.make_mask(end_path)
"""
