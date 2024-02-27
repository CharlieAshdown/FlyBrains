import torch
import torchvision
import torchvision.transforms as transforms

from os import listdir
from os.path import isfile, join

import cv2

training_set_path = "C:/Users/Charlie/Documents/samples/samples_26_02_2024/test_17_cropped/"

onlyfiles = [f for f in listdir(training_set_path) if isfile(join(training_set_path, f))]

training_data = {}

for path in onlyfiles:
    image = cv2.imread(path)
    training_data[path] = image

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])