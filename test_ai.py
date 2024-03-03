import matplotlib.pyplot as plt
import torch
import os
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from image_recognition_ai import get_transform

model_path = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_3_model"
root = "C:/Users/Charlie/Documents/samples/samples_29_02_2024/training/test_3_training"
model = torch.load(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Test model
image = read_image(os.path.join(root, "images/000.png"))
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

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))