from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
yolo = YOLO('pretrain/round2model.pt')

# Read the image
image = cv2.imread('image_0000.png')

# Run segmentation
results = yolo(image, task='segment')

# Iterate over each mask in the results
for i, mask in enumerate(results[0].masks.data):  # Accessing the actual mask data
    mask_np = mask.cpu().numpy() * 255  # Convert to NumPy and scale
    mask_image = np.uint8(mask_np)  # Convert to uint8 for displaying
    cv2.imshow(f'mask_{i}', mask_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
