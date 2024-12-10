from CEEC_Library import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time
import os
import torch

from ultralytics import YOLO

from tools.custom import LandDetect
from tools.controller import Controller
from utils.config import ModelConfig, ControlConfig
from utils.socket import create_socket
from tools.segmentation import myGetSegment, filter_masks_by_confidence

if __name__ == "__main__":
    # Create socket
    # Config model
    config_model = ModelConfig()
    device = torch.device('cuda')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Config control
    config_control = ControlConfig()

    # Controller
    controller = Controller()

    # Load YOLOv8
    yolo = YOLO(config_model.weights_yolo)

    # Load PIDNet
    image_save_folder = "training_images"
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    image_counter = 0

    # Set save frequency (Save every N frames)
    save_frequency = 1  # Save every 10 frames
    frame_counter = 0     # Initialize frame counter

    try:
        cnt_fps = 0
        t_pre = 0

        # Mask segmented image
        mask_lr = False
        mask_l = False
        mask_r = False
        mask_t = False

        # Counter for speed up after turning
        reset_counter = 0

        while True:
            s = GetStatus()

            try:
                img = GetRaw()

                # Increment frame counter
                frame_counter += 1
                img = cv2.resize(img, (640, 384))
                # Save image only if frame_counter reaches the save frequency
                if frame_counter % save_frequency == 0:
                    image_filename = os.path.join(image_save_folder, f"image_{image_counter:04d}.png")
                    cv2.imwrite(image_filename, img)
                    print(f"Saved image: {image_filename}")
                    image_counter += 1  # Increment image counter

                # segmented_image = myGetSegment(img, yolo)
                # segmented_image = cv2.resize(segmented_image, (320, 180))

                # if True:
                #     error = controller.calc_error(segmented_image)
                #     angle = controller.PID(error, p=0.2, i=0.0, d=0.02)
                #     # Speed up after turning (in 35 frames)
                #     if reset_counter >= 1 and reset_counter < 35:
                #         speed = 50
                #         reset_counter += 1
                #     elif reset_counter == 35:
                #         reset_counter = 0
                #         speed = 50 
                #     else:
                #         speed = controller.calc_speed(angle)

                #         if float(config_control.current_speed) > 44.5:
                #             speed = 15

                #     print("Error:", error)
                #     print("Angle:", angle)
                #     print("Speed:", speed)

                #     config_control.update(-angle, speed)
                
                # AVControl(speed = speed, angle = -angle)

                # Show image
                cv2.imshow("segmented image", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                # Calculate FPS
                if cnt_fps >= 90:
                    t_cur = time.time()
                    fps = (cnt_fps + 1) / (t_cur - t_pre)
                    t_pre = t_cur
                    print('FPS: {:.2f}\r\n'.format(fps))
                    cnt_fps = 0

                cnt_fps += 1

            except Exception as er:
                pass

    finally:
        print('closing socket')
        CloseSocket()
