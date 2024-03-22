import unittest
from utilities import video_converter, file_combiner, NumpyQueue
import numpy as np

class UtilitiesTest(unittest.TestCase):
    def test_video_converter(self):
        video_path = "C:/Users/Charlie/Documents/samples/samples_11_03_2024/videos/test_2.h264"
        frames_path = "C:/Users/Charlie/Documents/samples/samples_11_03_2024/videos/test_2_frames/"
        video_converter(video_path, frames_path)

    def test_file_combiner(self):
        images_roots = ["C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_1_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_2_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_3_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_4_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_5_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_6_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_7_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_8_training/images/",
                        "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_9_training/images/"]

        all_images = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/all_training/images/"

        file_combiner(images_roots, all_images)

    def test_numpy_queue(self):
        queue = NumpyQueue(max_size=3)

        a = [[1, 2], [3, 4]]
        queue.put(a)

        a = [[1, 2], [3, 4]]
        queue.put(a)

        a = [[1, 2], [3, 4]]
        queue.put(a)

        mean = queue.mean()
        distance = queue.speed()


if __name__ == '__main__':
    unittest.main()
