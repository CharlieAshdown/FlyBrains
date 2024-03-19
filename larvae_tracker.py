import numpy as np
import shutil

from sort import *
import matplotlib.pyplot as plt
import torch
import time
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, save_image
from image_recognition_ai import get_transform
from os.path import isfile, join, splitext

from utilities import video_converter, video_maker


class LarvaeTracker:
    def __init__(self, model, parent_dir, save_dir):
        self.model = model
        self.parent_dir = parent_dir
        self.save_dir = save_dir
        self.directory = "temp_frames"
        self.output = "temp_bounded_frames"

    def track_video(self, video_name, display=False):
        """
        Tracks the flies in the video.
        :param video_name: Name of the video to be processed
        :param display: Whether to display the frames or not
        :return:
        """

        model = self.model
        parent_dir = self.parent_dir
        save_dir = self.save_dir
        directory = self.directory
        output = self.output

        # Path
        frames_path = os.path.join(parent_dir, directory)
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        output_path = os.path.join(parent_dir, output)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        fps = video_converter(os.path.join(parent_dir, video_name), frames_path + "/")

        tracker = Sort(iou_threshold=0.1)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        colours = np.random.rand(32, 3)  # used only for display

        if display:
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(111, aspect='equal')

        image_paths = list(sorted(os.listdir(frames_path)))
        for image_path in image_paths:
            image = read_image(frames_path + "/" + image_path)
            eval_transform = get_transform(train=False)

            model.eval()
            with torch.no_grad():
                x = eval_transform(image)
                # convert RGBA -> RGB and move to device
                x = x[:3, ...].to(device)
                predictions = model([x, ])
                pred = predictions[0]

            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
            image = image[:3, ...]
            pred_labels = [f"larvae: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
            pred_boxes = pred["boxes"].long()
            output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

            masks = (pred["masks"] > 0.7).squeeze(1)
            output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

            if display:
                ax1.imshow(output_image.permute(1, 2, 0))
                plt.title(splitext(image_path)[0] + ' Tracked Targets')

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            # boxes = boxes[scores > 0.7]
            # scores = scores[scores > 0.7]

            detections = np.column_stack((boxes, scores))
            track_larvae = tracker.update(detections)
            boxes = []
            colour = []
            for d in track_larvae:
                d = d.astype(np.int32)
                if display:
                    ax1.add_patch(
                        patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                boxes.append([d[0], d[1], d[2], d[3]])
                colour.append(tuple((colours[d[4] % 32, :]*255).astype(np.int32)))
            boxes = torch.tensor(boxes)
            output_image = draw_bounding_boxes(output_image, boxes, colors=colour, width=3)

            save_image(output_image.float(), os.path.join(output_path + "/", image_path), normalize=True)
            if display:
                fig.canvas.flush_events()
                plt.draw()
                # plt.show()
                ax1.cla()

        video_maker(os.path.join(save_dir, splitext(video_name)[0] + "_tracked" + ".mp4"), output_path + "/", fps)
        shutil.rmtree(frames_path)
        shutil.rmtree(output_path)
        print("done")


if __name__ == "__main__":
    video_name = "test_12.h264"
    parent_dir = "/larvae_tracking_videos/unprocessed_video/"
    save_dir = "/larvae_tracking_videos/processed_video/"

    if torch.cuda.is_available():
        model_path = "/ai_models/model_gpu.pth"
    else:
        model_path = "/ai_models/model_cpu.pth"

    model = torch.load(model_path)

    larvae_tracker = LarvaeTracker(model, parent_dir, save_dir)
    larvae_tracker.track_video(video_name)