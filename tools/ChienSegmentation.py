from ultralytics import YOLO
import cv2
import numpy as np
def filter_masks_by_confidence(results, confidence_threshold=0.5):
    """
    Loại bỏ các bounding boxes có mức độ tự tin < 0.5 và trả về các mask tương ứng.
    :param results: Kết quả từ mô hình YOLOv8 sau khi thực hiện inference
    :param confidence_threshold: Ngưỡng mức độ tự tin để giữ lại (mặc định là 0.5)
    :return: Danh sách các mask đã lọc
    """
    filtered_masks = []

    # Lặp qua từng bounding box và confidence
    for i, box in enumerate(results[0].boxes):
        confidence = box.conf.item()  # Lấy mức độ tự tin của bounding box

        # Kiểm tra nếu mức độ tự tin >= ngưỡng
        if confidence >= confidence_threshold:
            filtered_masks.append(results[0].masks.data[i])  # Thêm mask tương ứng vào danh sách

    return filtered_masks
def myGetSegment(image_add, yolo_model_add):
  model = YOLO(yolo_model_add)
  image = cv2.imread(image_add)
  results = model(image)
  for result in results:
      for j, mask in enumerate(result['masks']):
          mask_np = mask.cpu().numpy() * 255
          mask_image = np.uint8(mask_np)
          mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
          mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
          result['masks'][j] = mask_image
#   filtered_masks = filter_masks_by_confidence(results)
#   for i, mask in enumerate(filtered_masks):
#     mask_np = mask.cpu().numpy() * 255  # Chuyển đổi mask sang định dạng numpy và nhân với 255
#     mask_image = np.uint8(mask_np)
#      # Chuyển đổi sang định dạng uint8
    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # Chuyển đổi sang định dạng BGR
    # mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))  # Thay đổi kích thước mask
    # #Đổi màu segment trong mask sang RGB(100,10,100)
    # custom_color = (128, 64, 128)
    # colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    # colored_mask[mask_np > 0] = custom_color  # Tô màu cho các pixel thuộc đối tượng
  return mask_image
