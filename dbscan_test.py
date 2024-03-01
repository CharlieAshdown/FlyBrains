import time

from image_processing import ImageProcessing

start_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/test_2"
end_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/test_2_mask"

processing = ImageProcessing(samples_path=start_path, epsilon=10, min_samples=5)
start_time = time.time()
processing.make_mask(end_path)
end_time = time.time()

print("Mask making run time: " + str(end_time-start_time))

"""
processing.set_dbscan(new_epsilon=5)
processing.make_mask(end_path)

processing.set_dbscan(new_epsilon=5)
processing.make_mask(end_path)
"""
