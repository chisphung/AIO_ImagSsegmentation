from ultralytics import YOLO
import cv2
import numpy as np
def filter_masks_by_confidence(results, confidence_threshold=0.0):
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
    mask_np = mask.cpu().numpy() * 255  # Chuyển đổi mask sang định dạng numpy và nhân với 255
    mask_image = np.uint8(mask_np)

     # Chuyển đổi sang định dạng uint8
    #mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)  # Chuyển đổi sang định dạng BGR
    #mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))  # Thay đổi kích thước mask
    #Đổi màu segment trong mask sang RGB(100,10,100)
    #custom_color = (255, 255, 255)
    #colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    #colored_mask[mask_np > 0] = custom_color  # Tô màu cho các pixel thuộc đối tượng
  return mask_image