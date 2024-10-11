from CEEC_Library import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import os
import torch

from ultralytics import YOLO

from tools.custom import LandDetect
from tools.controller1 import Controller
from utils.config import ModelConfig, ControlConfig
from utils.socket import create_socket
from tools.chienSegmentation import myGetSegment, filter_masks_by_confidence

if __name__ == "__main__":
    # Create socket
    # Config model
    config_model = ModelConfig()
    # device = torch.device('cuda')
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # half = device.type != 'cpu'

    # Config control
    config_control = ControlConfig()

    # Controller
    controller = Controller()

    # Load YOLOv8
    yolo = YOLO(config_model.weights_yolo)
    print("READY: YOLO loaded nhung ma please for the first frame to be processed")

    # Load PIDNet
    # image_save_folder = "training_images"
    # if not os.path.exists(image_save_folder):
    #     os.makedirs(image_save_folder)
    # image_counter = 0

    #land_detector = LandDetect('pidnet-s', os.path.join(config_model.weights_lane))
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
            """
            Input:
                image: the image returned from the car
                current_speed: the current speed of the car
                current_angle: the current steering angle of the car
            You must use these input values to calculate and assign the steering angle and speed of the car to 2 variables:
            Control variables: sendBack_angle, sendBack_Speed
                where:
                sendBack_angle (steering angle)
                sendBack_Speed (speed control)
            """
            
            # try:
            
            


            try:
                # =============Use our segmentation============

                img = GetRaw() 
                  # Get YOLO model output
                segmented_image = myGetSegment(img, yolo)
                # Debugging: Print out bounding boxes and classes
                segmented_image = cv2.resize(segmented_image, (320, 180))
                #segmented_image = segmented_image[5:6, 312:174]

                # =============Use BTC segmentation============
                # segmented_image = GetSeg()
                # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
                # segmented_image = (segmented_image*(255/np.max(segmented_image))).astype(np.uint8)


                if True:
                    error = controller.calc_error(segmented_image, height= 108)
                    
                    angle = controller.PID(error, p=0.15,  i=0.0, d=0.06)
                    #Speed up after turning (in 35 frames)

                    if reset_counter >= 1 and reset_counter < 35:
                        speed = 50
                        reset_counter += 1
                    elif reset_counter == 35:
                        reset_counter = 0
                        speed = 50 
                    else:
                        if (error == -1):
                            speed = 30
                        # elif (forsee_angle >20 ):
                        #     speed = 10
                        else:
                            speed = controller.calc_speed(angle)

                            if float(config_control.current_speed) > 44.5:
                                speed = 15

                    print("Error:", error)
                    print("Angle:", angle)
                    print("Speed:", speed)
                    config_control.update(-angle, speed)
                
                AVControl(speed = speed, angle = -angle)
                cv2.imshow("raw image", img)
                cv2.imshow("segmented image", segmented_image)
                # if config_model.view_first_view:
                #     cv2.imshow("first view image", yolo_output)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                # ============================================================ Calculate FPS
                if cnt_fps >= 90:
                    t_cur = time.time()
                    fps = (cnt_fps + 1)/(t_cur - t_pre)
                    t_pre = t_cur
                    print('FPS: {:.2f}\r\n'.format(fps))
                    cnt_fps = 0

                cnt_fps += 1

            except Exception as er:
                pass

    finally:
        print('closing socket')
        CloseSocket()