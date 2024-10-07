from ultralytics import YOLO
import cv2
import numpy as np
def filter_masks_by_confidence(results, confidence_threshold=0.5):
    filtered_masks = []

    for i, box in enumerate(results[0].boxes):
        confidence = box.conf.item()  
        if confidence >= confidence_threshold:
            filtered_masks.append(results[0].masks.data[i])  
    return filtered_masks

def myGetSegment(image, yolo):
  results = yolo(image, task='segment')
  filtered_masks = filter_masks_by_confidence(results)
  for i, mask in enumerate(filtered_masks):
    mask_np = mask.cpu().numpy() * 255  
    mask_image = np.uint8(mask_np)
  return mask_image