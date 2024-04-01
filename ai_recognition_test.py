import shutil
import unittest
import matplotlib.pyplot as plt
import torch
import time
from utilities import video_converter
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from image_recognition_ai import get_transform
import os


class ImageRecognitionTest(unittest.TestCase):

    def test_one_image(self):
        """
        Tests the image recognition AI on one image
        :return:
        """
        model_path = "ai_models/model_gpu.pth"
        root = "ai_training_data/processed_data/test_2_training/images"
        model = torch.load(model_path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Test model
        image = read_image(root + "/000.png")
        start = time.time()
        eval_transform = get_transform(train=False)

        model.eval()
        with torch.no_grad():
            x = eval_transform(image)
            # convert RGBA -> RGB and move to device
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]
        end = time.time()
        print("Time taken: " + str(end-start))
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        pred_labels = [f"larvae: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
        pred_boxes = pred["boxes"].long()
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        masks = (pred["masks"] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))
        plt.show()

    def test_video(self):
        display = False
        directory = "temp_frames"
        if torch.cuda.is_available():
            model_path = "ai_models/model_gpu.pth"
        else:
            model_path = "ai_models/model_cpu.pth"

        root = "larvae_tracking_videos/unprocessed_video/"

        frames_path = os.path.join(root, directory)
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        model = torch.load(model_path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Test model
        f = open("test_results/video_test_4_num_larvae_ai_test.txt", "w")
        f.write("frame_num\tai_num\thuman_num\n")
        fps = video_converter(root + "test_4.h264", frames_path + "/")
        image_paths = list(sorted(os.listdir(frames_path + "/")))

        eval_transform = get_transform(train=False)

        model.eval()
        for frame_num, image_path in enumerate(image_paths):
            start = time.time()
            image = read_image(frames_path + "/" + image_path)
            with torch.no_grad():
                x = eval_transform(image)
                # convert RGBA -> RGB and move to device
                x = x[:3, ...].to(device)
                predictions = model([x, ])
                pred = predictions[0]
            end = time.time()

            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
            image = image[:3, ...]
            pred_scores = pred["scores"][pred["scores"] > 0.9]
            labels = pred["labels"][pred["scores"] > 0.9]
            pred_labels = [f"larvae: {score:.3f}" for label, score in zip(labels, pred_scores)]
            pred_boxes = pred["boxes"].long()
            ai_num = len(pred_boxes)
            pred_boxes = pred_boxes[pred["scores"] > 0.9]
            output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

            masks = (pred["masks"] > 0.7).squeeze(1)
            masks = masks[pred["scores"] > 0.9]
            output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")
            if display:
                print("Time taken: " + str(end - start))
                plt.figure(figsize=(12, 12))
                plt.imshow(output_image.permute(1, 2, 0))
                plt.show()
                human_larvae = input("Input number of larvae")
            else:
                human_larvae = 10
            f.write(f"{frame_num}\t{ai_num}\t{human_larvae}\n")

        f.close()
        shutil.rmtree(frames_path)


if __name__ == '__main__':
    unittest.main()