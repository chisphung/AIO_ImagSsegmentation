import numpy as np
import time
import cv2
from utils.utils import find_majority


class Controller():
    def __init__(self):
        # Initialize variables for PID control and traffic signs detection
        self.error_arr = np.zeros(5)       # Array to store the error values for PID control
        self.error_sp = np.zeros(5)        # Array to store the speed error values for PID control

        self.pre_t = time.time()           # Store the current time for steering control
        self.pre_t_spd = time.time()       # Store the current time for speed control

        self.sendBack_angle = 0            # Initialize the steering angle to 0
        self.sendBack_speed = 0            # Initialize the speed to 0

        # List of all the possible traffic light labels
        self.traffic_lights = ['turn_right', 'turn_left', 'straight', 'no_turn_left', 'no_turn_right', 'no_straight']
        
        # List of all the possible object detection labels
        self.class_names = ['no', 'turn_right', 'straight', 'no_turn_left', 'no_turn_right', 'no_straight', 'car', 'unknown', 'turn_left']
        

        self.stored_class_names = []       # List to store the detected labels for finding majority class

        self.majority_class = ""           # Initialize the majority class to empty
        self.start_cal_area = False        # Flag to start calculating the area for turning
        self.turning_counter = 0           # Counter to track the number of turning frames
        self.angle_turning = 0             # Angle to turn the car

        self.sum_left_corner = 0           # Sum of the pixel values in the left corner of the image
        self.sum_right_corner = 0          # Sum of the pixel values in the right corner of the image
        self.sum_top_corner = 0            # Sum of the pixel values in the top corner of the image

        self.mask_l = False                # Flag to indicate mask left image
        self.mask_r = False                # Flag to indicate mask right image
        self.mask_lr = False               # Flag to indicate mask leftn and right image
        self.mask_t = False                # Flag to indicate mask top image

        self.next_step = False             # Flag to indicate the next step of the car when turning
        self.is_turning = False            # Flag to indicate if the car is currently turning

        self.reset_counter = 0             # Counter to track the number of frames since the last reset

        self.is_turn_left = False          # Flag to indicate if the car is turning left
        self.is_turn_right = False         # Flag to indicate if the car is turning right
        self.is_straight = False           # Flag to indicate if the car is going straight

        self.is_no_turn_right_case_1 = False    # Flag to indicate if there is a "no turn right" sign in case 1
        self.is_no_turn_right_case_2 = False    # Flag for case 2
        self.is_no_turn_right_case_3 = False    # Flag for case 3
        self.is_no_turn_right_case_4 = False    # Flag for case 4

        self.is_turn_left_case_1 = False        # Flag to indicate if there is a "turn left" sign in case 1
        self.is_turn_left_case_2 = False        # Flag for case 2

    def reset(self):
        # Reset all values to default values
        self.turning_counter = 0
        self.majority_class = ""
        self.start_cal_area = False
        self.stored_class_names = []

        self.mask_lr = False
        self.mask_l = False
        self.mask_r = False
        self.mask_t = False

        self.majority_class = ""
        self.start_cal_area = False
        self.turning_counter = 0
        self.angle_turning = 0

        self.next_step = False
        self.is_turning = False

        self.reset_counter = 0

        self.is_turn_left = False
        self.is_turn_right = False
        self.is_straight = False

        self.is_no_turn_right_case_1 = False
        self.is_no_turn_right_case_2 = False
        self.is_no_turn_right_case_3 = False
        self.is_no_turn_right_case_4 = False

        self.is_turn_left_case_1 = False


    # def showLine(self, image, height, height1):
    #     img = image.copy()
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     cv2.line(img, (0, height), (img.shape[1], height), (255, 0, 0), 1)
    #     cv2.line(img, (0, height1), (img.shape[1], height1), (255, 0, 0), 1)
    #     cv2.imshow("line", img)

    def calc_error(self, image, height):
        """
        Calculates the error between the center of the right lane and the center of the image.

        Args:
        image: A NumPy array representing the image.

        Returns:
        The error between the center of the right lane and the center of the image.
        """

        arr = []
        # height = 60
        #height = 40
        #height = 113
        # height = 
    
        #self.showLine(image, height, height)
        
        lineRow = image[height, :]
        flag = -1
        try:
            for x, y in enumerate(lineRow):
                if y == 255:
                    flag = x  
                    break          
            if flag != -1:
                for x in range(flag, len(lineRow)):
                    if lineRow[x] == 255:
                        arr.append(x)  # Append x to arr for consecutive '255'
                    else:
                        break  # Stop when a '0' (non-white) pixel is found
            # print("min", min(arr))
            # print("max", max(arr))
            if(max(arr) - min(arr) > 200 and min(arr) < 150):
                print("Intersection detected")
                return -1
            # #if(max(arr) - min(arr) > 230):
            #     return 0
            if len(arr) > 0:
                center_right_lane = int((min(arr) + max(arr)*2.5)/3.5) - 10
                error = int(image.shape[1]/2) - center_right_lane
                return error*1.3
            else:
                return 0
        except:
            return 0


    def PID(self, error, p, i, d):
        """
        Calculates the PID output for the specified error.

        Args:
        error: The error value.
        p: The proportional gain.
        i: The integral gain.
        d: The derivative gain.

        Returns:
        The PID output.
        """
        if error == -1:
              error = 0
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.pre_t
        self.pre_t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D

        if abs(angle) > 25:
            angle = np.sign(angle)*25

        return int(angle)



    def calc_speed(self, angle):
        """
        Calculates the speed of the car based on the steering angle.

        Args:
        angle: The steering angle.

        Returns:
        The speed of the car.
        """
        if abs(angle) < 10:
            speed = 50
        elif 10 <= abs(angle) <= 18:
            speed = 1
        elif 18 < abs(angle) <= 20:
            speed = 0
        elif 20 < abs(angle) <= 25:
            speed = -5
        else:
            speed = -1
        return speed

