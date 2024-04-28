import shutil

import numpy as np
import torch
import csv
import warnings
import cv2

from skimage.morphology import skeletonize
from sklearn import decomposition
from sort import *
from os.path import splitext
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, save_image
from torchvision.transforms import Pad, CenterCrop
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from image_recognition_ai import get_transform
from utilities import video_converter, video_maker, Queue, new_set, get_led_timings


class LarvaeTracker:
    def __init__(self, model, parent_dir, save_dir, csv_write=False):
        self.model = model
        self.parent_dir = parent_dir
        self.save_dir = save_dir
        self.directory = "temp_frames"
        self.output = "temp_bounded_frames"
        self.video_path = None
        self.led_on, self.led_off = None, None
        self.csv_write = csv_write
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        self.pixels_per_mm = 26

    def track_video(self, video_name, array_len=10, accuracy=0.9, display=False, pad_amount=10, save_video=True):
        """
        Tracks the flies in the video.
        :param video_name: Name of the video to be processed
        :param array_len: Length of tracking array
        :param accuracy: The prediction accuracy threshold to accept a larva
        :param display: Whether to display the frames or not
        :param pad_amount: The size of the video border
        :param save_video: Whether to save the final video (change to False if testing)
        :return:
        """

        model = self.model
        parent_dir = self.parent_dir
        save_dir = self.save_dir
        directory = self.directory
        output = self.output

        if self.csv_write:
            fields = ["time", "larvae", "speed (mm/s)", "rotation_speed (degree/s)", "is_led_on", "has_led_been_on"]
            num_larvae_fields = ["time", "num_larvae"]
            with open(f"{splitext(video_name)[0]}.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
            with open(f"{splitext(video_name)[0]}_larvae_count_{int(accuracy*100)}_accuracy.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=num_larvae_fields)
                writer.writeheader()

        # Path
        frames_path = os.path.join(parent_dir, directory)
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        output_path = os.path.join(parent_dir, output)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        fps = video_converter(os.path.join(parent_dir, video_name), frames_path + "/")
        self.led_on, self.led_off = get_led_timings(self.parent_dir, fps)
        tracker = Sort(iou_threshold=0.7)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        colours = np.random.rand(32, 3)

        if display:
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(111, aspect='equal')

        image_paths = list(sorted(os.listdir(frames_path)))
        distances = Queue(max_size=array_len)
        rotation_angles = Queue(max_size=array_len)
        max_len = 32
        for image_num, image_path in enumerate(image_paths):
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
            pred_boxes = pred["boxes"].long()
            output_image = draw_bounding_boxes(image, pred_boxes, colors="red")

            masks = (pred["masks"] > 0).squeeze(1)
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            boxes = boxes[scores > accuracy]
            masks = masks[scores > accuracy, ...]
            scores = scores[scores > accuracy]

            output_image = draw_segmentation_masks(output_image, (pred["masks"] > 0.7).squeeze(1), alpha=0.5, colors="blue")

            if display:
                ax1.imshow(output_image.permute(1, 2, 0))
                plt.title(splitext(image_path)[0] + ' Tracked Targets')

            detections = np.column_stack((boxes, scores))
            numpy_masks = (masks.cpu().numpy() * 255).round().astype(np.uint8)
            mask_ids = np.arange(0, len(numpy_masks))

            track_larvae, mask_ids = tracker.update(dets=detections, mask_ids=mask_ids)
            boxes = []
            if len(track_larvae) > max_len:
                max_len = len(track_larvae)
            centers = list(np.full((max_len, 2), np.nan))
            colour = []
            for d in track_larvae:
                d = d.astype(np.int32)
                if display:
                    ax1.add_patch(
                        patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                boxes.append([d[0], d[1], d[2], d[3]])
                centers[d[4] % max_len] = [(d[2] - d[0])/2, (d[3] - d[1])/2]
                colour.append(tuple((colours[d[4] % 32, :]*255).astype(np.int32)))
            distance = [np.sqrt(x*x + y*y) for x, y in centers]
            boxes = torch.tensor(boxes)
            distances.put(distance)
            speeds = distances.speed()
            speeds = [x / self.pixels_per_mm * fps for x in speeds]
            speeds_ordered = []
            labels = []

            angles = self._get_angle_pca(numpy_masks[mask_ids])
            ordered_angles = list(np.full((max_len, 1), np.nan).squeeze(1))
            for i, mask_id in enumerate(mask_ids):
                ordered_angles[mask_id] = angles[i]
            rotation_angles.put(ordered_angles)
            rotation_speeds = rotation_angles.speed()
            rotation_speeds = [x * fps for x in rotation_speeds]
            rotation_speeds = rotation_speeds[:np.max(mask_ids)+1]

            for d in track_larvae:
                speeds_ordered.append(speeds[round(d[4] % max_len)])
                labels.append(d[4])

            if self.csv_write:
                data_lines = []
                if not self.led_on:
                    is_led_on = 0
                    has_led_been_on = 0
                elif self.led_on <= image_num <= self.led_off:
                    is_led_on = 1
                    has_led_been_on = 1
                elif self.led_on <= image_num:
                    is_led_on = 0
                    has_led_been_on = 1
                else:
                    is_led_on = 0
                    has_led_been_on = 0
                for label, speed, rotation_speed in zip(labels, speeds_ordered, rotation_speeds):
                    data_lines.append({"time": f"{(image_num/fps):.3f}",
                                       "larvae": label,
                                       "speed": f"{speed:.3f} mm/s",
                                       "rotation_speed": f"{rotation_speed:.3f} degrees/second",
                                       "is_led_on": is_led_on,
                                       "has_led_been_on": has_led_been_on})
                with open(f"{splitext(video_name)[0]}.csv", 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writerows(data_lines)

            labels = [f"larvae: {label} \n speed: {speed:.3f} \n rotation_speed: {rotation_speed:.3f}" for
                      label, speed, rotation_speed in zip(labels, speeds_ordered, rotation_speeds)]

            if self.csv_write:
                with open(f"{splitext(video_name)[0]}_larvae_count_{int(accuracy*100)}_accuracy.csv", 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=num_larvae_fields)
                    writer.writerow({"time": f"{(image_num/fps):.3f}", "num_larvae": len(labels)})

            output_image = draw_bounding_boxes(output_image, boxes, labels, colors=colour, width=3)

            if self.led_on is not None and self.led_on <= image_num <= self.led_off:
                output_image = to_pil_image(output_image)
                crop = CenterCrop((new_set(output_image.size) - 20))
                output_image = crop(output_image)
                pad = Pad(pad_amount, fill=(255, 0, 0), padding_mode="constant")
                output_image = pad(output_image)
                output_image = pil_to_tensor(output_image)

            save_image(output_image.float(), os.path.join(output_path + "/", image_path), normalize=True)
            if display:
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()
        if save_video:
            video_maker(os.path.join(save_dir, splitext(video_name)[0] + "_tracked" + ".mp4"), output_path + "/", fps)
            self.video_path = os.path.join(save_dir, splitext(video_name)[0] + "_tracked" + ".mp4")
        shutil.rmtree(frames_path)
        shutil.rmtree(output_path)
        print("done")

    def get_video_path(self):
        return self.video_path

    def _get_angles(self, masks, num_splits=2):
        """
        Gets the angles of rotation for the flies
        :param num_splits:
        :param masks:
        :return:
        """
        display = False
        angles = []
        for mask in masks:
            if display:
                cv2.imshow("mask", mask)
                cv2.waitKey(0)

            skeleton_mask = skeletonize(mask)
            skeleton_mask_coords = np.argwhere(skeleton_mask > 0)

            # PCA
            pca = decomposition.PCA(n_components=2)
            pca.fit(skeleton_mask_coords)
            rotated_skeleton_mask_coords = pca.fit_transform(skeleton_mask_coords)

            skeleton_x, skeleton_y = np.split(rotated_skeleton_mask_coords, [-1], axis=1)

            # Reduce the axis
            skeleton_x = skeleton_x.squeeze(1)
            skeleton_y = skeleton_y.squeeze(1)

            skeleton_x_split = np.array_split(skeleton_x, num_splits)
            skeleton_y_split = np.array_split(skeleton_y, num_splits)
            angle = 0
            for split in range(len(skeleton_x_split)):
                M2 = np.polyfit(skeleton_x_split[split], skeleton_y_split[split], 1)[0]
                angle += (abs(np.arctan(M2)) * 180 / np.pi)
            angle = 180 - angle
            angles.append(angle)
        return np.array(angles)

    def _get_angle_pca(self, masks):
        """
        Determines the angle of rotation of the larvae using PCA
        :param masks: Array of masks to find angle from
        :return: Array containing the angle of rotation for each mask
        """
        display = False
        length_diff = []
        for mask in masks:
            if display:
                cv2.imshow("mask", mask)
                cv2.waitKey(0)

            skeleton_mask = skeletonize(mask)
            skeleton_mask_coords = np.argwhere(skeleton_mask > 0)

            # PCA
            pca = decomposition.PCA(n_components=2)
            pca.fit(skeleton_mask_coords)
            rotated_skeleton_mask_coords = pca.fit_transform(skeleton_mask_coords)

            skeleton_x, skeleton_y = np.split(rotated_skeleton_mask_coords, [-1], axis=1)

            skeleton_x = skeleton_x.squeeze(1)
            skeleton_y = skeleton_y.squeeze(1)

            end_to_end = np.max(skeleton_x) - np.min(skeleton_x)
            length = np.max(skeleton_y) - np.min(skeleton_y)
            length_diff.append(np.arctan(length/end_to_end) * 180 / np.pi)
        return length_diff


if __name__ == "__main__":
    if torch.cuda.is_available():
        model_path = "ai_models/model_gpu.pth"
    else:
        model_path = "ai_models/model_cpu.pth"

    tests = ["test_002", "test_003", "test_004", "test_005", "test_006",
             "test_007", "test_008", "test_009", "test_010", "test_011",
             "test_012"]

    # test_number = "test_010"
    for test_number in tests:
        root = "D:/Flybrains/samples/samples_11_03_2024/videos"
        parent_dir = f"{root}/{test_number}"
        save_dir = parent_dir
        video_name = glob.glob(f"{parent_dir}/*.h264")[0]

        model = torch.load(model_path)

        larvae_tracker = LarvaeTracker(model, parent_dir, save_dir, csv_write=True)
        larvae_tracker.track_video(video_name, array_len=10, accuracy=0.9, save_video=False)
