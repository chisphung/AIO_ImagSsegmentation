import numpy as np
import time

from utils.utils import find_majority


class Controller():
    def __init__(self):
        # Initialize variables for PID control and traffic signs detection
        # Array to store the error values for PID control
        self.error_arr = np.zeros(5)
        # Array to store the speed error values for PID control
        self.error_sp = np.zeros(5)

        # Store the current time for steering control timing
        self.pre_t = time.time()
        self.pre_t_spd = time.time()       # Store the current time for speed control

        self.sendBack_angle = 0            # Initialize the steering angle to 0
        self.sendBack_speed = 0            # Initialize the speed to 0

        # List of all the possible traffic light labels
        self.traffic_lights = ['left', 'no left', 'no right', 'right', 'stop', 'straight']

        # List of all the possible object detection labels
        # self.class_names = ['no right', 'stop',
        #                     'straight', 'left', 'right']
        self.class_names = ['left', 'no left', 'no right', 'right', 'stop', 'straight']

        # List to store the detected labels for finding majority class
        self.stored_class_names = []
        
        # New: Store detected signboards with their areas for weighted decision making
        self.signboard_history = []  # List of tuples: (label, area, timestamp)
        self.max_detected_area = 0   # Track maximum area detected for each signboard
        self.stored_signboard = None # The final signboard to use at intersection
        self.signboard_confidence = 0  # Weighted confidence score

        self.majority_class = ""           # Initialize the majority class to empty
        self.start_cal_area = False        # Flag to start calculating the area for turning
        self.turning_counter = 0           # Counter to track the number of turning frames
        self.angle_turning = 0             # Angle to turn the car

        # Sum of the pixel values in the left corner of the image
        self.sum_left_corner = 0
        # Sum of the pixel values in the right corner of the image
        self.sum_right_corner = 0
        # Sum of the pixel values in the top corner of the image
        self.sum_top_corner = 0

        self.mask_l = False                # Flag to indicate mask left image
        self.mask_r = False                # Flag to indicate mask right image
        self.mask_lr = False               # Flag to indicate mask leftn and right image
        self.mask_t = False                # Flag to indicate mask top image

        # Flag to indicate the next step of the car when turning
        self.next_step = False
        self.is_turning = False            # Flag to indicate if the car is currently turning

        # Counter to track the number of frames since the last reset
        self.reset_counter = 0

        self.is_turn_left = False          # Flag to indicate if the car is turning left
        self.is_turn_right = False         # Flag to indicate if the car is turning right
        self.is_straight = False           # Flag to indicate if the car is going straight

        # Flag to indicate if there is a "no turn right" sign in case 1
        self.is_no_turn_right_case_1 = False
        self.is_no_turn_right_case_2 = False    # Flag for case 2
        self.is_no_turn_right_case_3 = False    # Flag for case 3
        self.is_no_turn_right_case_4 = False    # Flag for case 4

        # Flag to indicate if there is a "turn left" sign in case 1
        self.is_turn_left_case_1 = False
        self.is_turn_left_case_2 = False        # Flag for case 2

        self.intersection_detected = False
        
        # NEW: State for "waiting for intersection after signboard confirmed"
        self.waiting_for_intersection = False

        # Lane tracking helpers for robustness in sharp turns
        self.last_center_right_lane = None
        self.lost_lane_frames = 0
        self.prev_error = 0.0

    def _safe_get_preds(self, yolo_output):
        """Return prediction array safely as numpy array with shape (N, >=6)."""
        try:
            if yolo_output is None or yolo_output.boxes is None or yolo_output.boxes.data is None:
                print("[_safe_get_preds] No valid yolo_output or boxes")
                return np.empty((0, 6), dtype=float)
            preds = yolo_output.boxes.data
            # Convert to cpu numpy if available (torch tensors)
            if hasattr(preds, "cpu"):
                preds = preds.cpu()
            if hasattr(preds, "numpy"):
                preds = preds.numpy()
            return preds if preds is not None else np.empty((0, 6), dtype=float)
        except Exception:
            return np.empty((0, 6), dtype=float)

    def _compute_region_sums(self, img):
        """Compute dynamic region sums used by logic, with bounds checks."""
        try:
            h, w = img.shape[0], img.shape[1]
        except Exception:
            return 0, 0, 0

        # Define window sizes with bounds
        left_w = min(24, max(1, w // 10))
        left_h = min(24, max(1, h // 10))
        right_w = min(50, max(1, w // 8))
        right_h = min(50, max(1, h // 8))
        top_h = min(24, max(1, h // 10))
        center_w = min(50, max(10, w // 5))

        # Left top corner
        sum_left = np.sum(img[:left_h, :left_w, 0]) if h > 0 and w > 0 else 0

        # Right top corner
        sum_right = np.sum(
            img[:right_h, max(0, w - right_w):, 0]) if w > 0 else 0

        # Top center window
        mid = w // 2
        half = center_w // 2
        l = max(0, mid - half)
        r = min(w, mid + half)
        sum_top = np.sum(img[:top_h, l:r, 0]) if r > l else 0

        return int(sum_left), int(sum_right), int(sum_top)

    def reset(self):
        # Reset all values to default values
        self.turning_counter = 0
        self.majority_class = ""
        self.start_cal_area = False
        self.stored_class_names = []
        
        # Reset signboard history and stored decision
        self.signboard_history = []
        self.max_detected_area = 0
        self.stored_signboard = None
        self.signboard_confidence = 0
        
        # Reset intersection waiting state
        self.waiting_for_intersection = False

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
        self.is_turn_left_case_2 = False

        self.intersection_detected = False

        # Reset lane tracking helpers
        self.last_center_right_lane = None
        self.lost_lane_frames = 0
        self.prev_error = 0.0

    def control(self, segmented_image, yolo_output):
        # Safety reset after many frames to avoid stale state
        if self.reset_counter >= 200:
            self.reset()

        # Calculate area of left, right, and top corner of the segmented image
        self.sum_left_corner, self.sum_right_corner, self.sum_top_corner = self._compute_region_sums(
            segmented_image)

        # Debug flags can be re-enabled if needed

        # ==================== NEW CLEAN STATE MACHINE ====================
        # State 1: Currently turning - execute turn sequence
        if self.is_turning:
            self.handle_turning()
        
        # State 2: Waiting for intersection after signboard confirmed
        elif self.waiting_for_intersection:
            print(f"[control] 🔍 Waiting for intersection... (signboard: {self.majority_class})")
            
            # Check if intersection detected (by calc_error or current frame)
            if self.intersection_detected or self.detect_intersection(segmented_image):
                print(f"[control] ✓ INTERSECTION REACHED! Applying signboard: {self.majority_class}")
                self.intersection_detected = True
                self.waiting_for_intersection = False
                
                # Apply the stored signboard decision
                self._apply_signboard_decision()
        
        # State 3: Collecting signboard samples (< 30 samples)
        elif len(self.stored_class_names) < 30:
            preds = self._safe_get_preds(yolo_output)
            
            for pred in preds:
                try:
                    class_id = int(pred[-1])
                except Exception:
                    continue
                if class_id < 0 or class_id >= len(self.class_names):
                    continue
                label = self.class_names[class_id]
                if label in self.traffic_lights:
                    self.stored_class_names.append(label)
                    print(f"[control] 📝 Added sign '{label}' to stored list (count={len(self.stored_class_names)})")
                    
                    # Store in signboard history with area for weighting
                    boxes = pred[:4]
                    try:
                        areas = float(max(0.0, (boxes[2] - boxes[0])) * max(0.0, (boxes[3] - boxes[1])))
                        current_time = time.time()
                        self.signboard_history.append((label, areas, current_time))
                        if areas > self.max_detected_area:
                            self.max_detected_area = areas
                        area_weight = (areas / 100.0) ** 1.5
                        self.signboard_confidence += area_weight
                        self.stored_signboard = label
                        print(f"[control] 💾 Stored: {label}, area={areas:.1f}, confidence={self.signboard_confidence:.1f}")
                    except Exception:
                        pass

                if label == 'left':
                    self.stored_class_names.extend(['left']*3)
                    print(f"[control] 📊 Boosted 'left' sign (count={len(self.stored_class_names)})")
            
            # # Check for early trigger: 10+ samples AND intersection already visible
            # if len(self.stored_class_names) >= 10 and (self.intersection_detected or self.detect_intersection(segmented_image)):
            #     print(f"[control] ⚡ EARLY TRIGGER: Intersection detected with {len(self.stored_class_names)} samples!")
            #     self.majority_class = find_majority(self.stored_class_names)[0]
            #     print(f"[control] 📋 Early majority class: {self.majority_class}")
            #     self.waiting_for_intersection = True  # Enter intersection waiting loop
                
        # State 4: 30+ samples collected, determine majority and start waiting for intersection
        elif len(self.stored_class_names) >= 30:
            # Determine the majority class from collected samples
            self.majority_class = find_majority(self.stored_class_names)[0]
            print(f"[control] ✅ Signboard confirmed: {self.majority_class} (from {len(self.stored_class_names)} samples)")
            print(f"[control] 🔍 Now waiting for intersection...")
            
            # Enter intersection waiting state
            self.waiting_for_intersection = True

        # Increment frame counter and return
        self.reset_counter += 1
        return self.sendBack_angle, self.sendBack_speed, self.next_step, self.mask_l, self.mask_r

    def _apply_signboard_decision(self):
        """
        Apply the stored majority_class signboard decision by setting turn parameters.
        This method is called when intersection is reached after signboard is confirmed.
        """
        print(f"[_apply_signboard_decision] Applying: {self.majority_class}")
        
        if self.majority_class == 'right':
            self.angle_turning = -25
            self.is_turning = True
            print("[_apply_signboard_decision] → RIGHT TURN (-25°)")
            
        elif self.majority_class == 'left':
            if self.sum_top_corner > 17_000:
                self.angle_turning = 32
                self.is_turn_left_case_1 = True
                print("[_apply_signboard_decision] → LEFT TURN Case 1 (32°)")
            else:
                self.angle_turning = 28
                self.is_turn_left_case_2 = True
                print("[_apply_signboard_decision] → LEFT TURN Case 2 (28°)")
            self.is_turning = True
            
        elif self.majority_class == 'straight':
            self.angle_turning = 0
            self.is_turning = True
            self.mask_l = True
            self.mask_r = True
            print("[_apply_signboard_decision] → STRAIGHT (0°, mask L+R)")
            
        elif self.majority_class == 'no right':
            if self.sum_left_corner > 2_000 and self.sum_top_corner < 17_500:
                self.angle_turning = 30
                self.is_no_turn_right_case_1 = True
                print("[_apply_signboard_decision] → NO RIGHT: Turn left (30°)")
            else:
                self.angle_turning = 0
                self.mask_l = True
                self.mask_r = True
                self.is_no_turn_right_case_2 = True
                print("[_apply_signboard_decision] → NO RIGHT: Go straight (mask L+R)")
            self.is_turning = True
            
        elif self.majority_class == 'stop':
            self.angle_turning = 0
            self.is_turning = True
            print("[_apply_signboard_decision] → STOP (0°)")
            
        elif self.majority_class == 'no left':
            if self.sum_right_corner > self.sum_top_corner/2:
                self.angle_turning = -25
                print("[_apply_signboard_decision] → NO LEFT: Turn right (-25°)")
            else:
                self.angle_turning = 0
                print("[_apply_signboard_decision] → NO LEFT: Go straight (0°)")
            self.is_turning = True
            
        elif self.majority_class == 'no_straight':
            if self.sum_left_corner > self.sum_right_corner*4:
                self.angle_turning = 26
                print("[_apply_signboard_decision] → NO STRAIGHT: Turn left (26°)")
            else:
                self.angle_turning = -24
                print("[_apply_signboard_decision] → NO STRAIGHT: Turn right (-24°)")
            self.is_turning = True
            
        else:
            print(f"[_apply_signboard_decision] ⚠️ Unhandled class: {self.majority_class}")

    def handle_turning(self):
        # Default config
        speed = 0
        angle = 0

        # Check turning counter
        MAX_COUNTER = 25
        if self.turning_counter < MAX_COUNTER:
            if self.majority_class == 'left':
                if self.is_turn_left_case_1:
                    speed = 0
                    if self.turning_counter <= 3:  # Hard
                        angle = 25
                    elif self.turning_counter > 3 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                elif self.is_turn_left_case_2:
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 5
                    elif self.turning_counter > 1 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'right':
                speed = 0
                # if self.turning_counter <= 1:
                #     angle = -5
                if self.turning_counter >= 0 and self.turning_counter < 7:
                    angle = self.angle_turning
                else:
                    self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'no left':
                speed = 0
                if self.turning_counter <= 1:
                    angle = 2
                elif self.turning_counter > 1 and self.turning_counter <= 5:
                    angle = -25
                else:
                    self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'no right':
                if self.is_no_turn_right_case_1:    # Left hard
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 5
                    elif self.turning_counter > 1 and self.turning_counter <= 7:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                elif self.is_no_turn_right_case_2:  # Left
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                elif self.is_no_turn_right_case_3:  # Straight (left of map)
                    speed = 0
                    if self.turning_counter <= 9:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                else:   # Straight: self.is_no_turn_right_case_4
                    speed = 0
                    if self.turning_counter <= 9:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'straight':
                speed = 0
                if self.turning_counter <= 9:
                    angle = self.angle_turning
                else:
                    self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'no_straight':
                if self.is_turn_left:  # Left
                    speed = 0
                    if self.turning_counter <= 11:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 5:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                else:  # Right
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 5:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
            elif self.majority_class == 'stop':

                if self.turning_counter <= 3:
                    speed = -25
                    angle = 0
                elif self.turning_counter > 3 and self.turning_counter <= 15:
                    speed = 0
                    angle = self.angle_turning
                else:
                    self.turning_counter = MAX_COUNTER

            elif self.angle_turning == 20:  # Forced left turn at intersection
                speed = 20  # Higher speed to pass quickly
                if self.turning_counter <= 30:  # Hold for longer
                    angle = self.angle_turning
                else:
                    self.turning_counter = MAX_COUNTER

            # Set default speed
            if speed == 0:
                speed = 30

            # Set send back values
            self.sendBack_angle = angle
            self.sendBack_speed = speed

            # Increase the counter by 1
            self.turning_counter += 1

            # Send back to not use PID calculate again when turning
            self.next_step = True

        elif self.turning_counter >= MAX_COUNTER:
            # Reset after turning
            self.reset()

    def handle_areas(self, areas, segmented_image):
        if areas < 100:
            self.reset()
        # if areas > 600.0 and self.majority_class == 'turn_right':
        if areas > 900.0 and self.majority_class == 'right':
            # CRITICAL: Check intersection BEFORE triggering turn!
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ RIGHT turn triggered at area={areas:.1f} with intersection confirmed")
                # Set angle and error turning
                self.angle_turning = -25
                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

        if areas > 800.0 and self.majority_class == 'left':
            # CRITICAL: Check intersection BEFORE triggering turn!
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ LEFT turn triggered at area={areas:.1f} with intersection confirmed")
                if self.sum_top_corner > 17_000:
                    self.is_turn_left_case_1 = True
                else:
                    self.is_turn_left_case_2 = True

                # Increase left turning strength
                self.angle_turning = 32 if self.is_turn_left_case_1 else 28

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

        if areas >= 400.0 and self.majority_class == 'no_turn_left':
            # CRITICAL: Check intersection BEFORE triggering
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ NO TURN LEFT triggered at area={areas:.1f} with intersection confirmed")
                if self.sum_right_corner > self.sum_top_corner/2:  # Turn right
                    angle = -25
                else:
                    angle = 0  # Straight

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False

                # Set global angle
                self.angle_turning = angle
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

        if areas >= 650.0 and self.majority_class == 'no right':
            # CRITICAL: Check intersection BEFORE triggering
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ NO RIGHT triggered at area={areas:.1f} with intersection confirmed")
                if (self.sum_left_corner > 2_000 and self.sum_top_corner < 17_500):  # \
                    #    or (self.sum_left_corner < 9_000 and self.sum_top_corner < 9_000):

                    if self.sum_top_corner < 3_000:  # Hard
                        self.is_no_turn_right_case_1 = True
                    else:
                        self.is_no_turn_right_case_2 = True

                    angle = 30  # Left

                elif self.sum_top_corner > 18_000:
                    self.is_no_turn_right_case_3 = True

                    angle = 0  # Straight
                    self.mask_l = True
                    self.mask_r = True

                else:
                    self.is_no_turn_right_case_4 = True

                    angle = 0  # Straight
                    self.mask_l = True
                    self.mask_r = True

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False

                # Set global angle
                self.angle_turning = angle
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

        if areas > 750.0 and self.majority_class == 'straight':
            # CRITICAL: Check intersection BEFORE triggering
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ STRAIGHT triggered at area={areas:.1f} with intersection confirmed")
                # Set global angle
                self.angle_turning = 0

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False

                self.mask_l = True
                self.mask_r = True
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

        if areas > 600.0 and self.majority_class == 'no_straight':
            # CRITICAL: Check intersection BEFORE triggering
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ NO STRAIGHT triggered at area={areas:.1f} with intersection confirmed")
                if self.sum_left_corner > self.sum_right_corner*4:
                    # Turn left
                    angle = 26
                    self.is_turn_left = True
                else:
                    # Turn right
                    angle = -24
                    self.is_turn_right = True

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False

                # Set angle and error turning
                self.angle_turning = angle
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")
                
        if areas > 380.0 and self.majority_class == 'stop':
            # CRITICAL: Check intersection BEFORE triggering
            if self.detect_intersection(segmented_image):
                print(f"[handle_areas] ✓ STOP triggered at area={areas:.1f} with intersection confirmed")
                # Set angle and error turning
                self.angle_turning = 0

                # Start turning and stop cal areas
                self.is_turning = True
                self.start_cal_area = False
            else:
                print(f"[handle_areas] ✗ Area={areas:.1f} reached but NO intersection detected, waiting...")

    def get_weighted_signboard_decision(self):
        """
        Calculate the most confident signboard from history based on area weighting.
        Returns the signboard label with highest weighted confidence.
        """
        if not self.signboard_history:
            return None
        
        # Aggregate weights by label
        label_weights = {}
        for label, area, timestamp in self.signboard_history:
            # Weight by area (larger = more confident) and recency
            area_weight = (area / 100.0) ** 1.5
            time_weight = 1.0  # Could add time decay if needed
            weight = area_weight * time_weight
            
            if label not in label_weights:
                label_weights[label] = 0
            label_weights[label] += weight
        
        # Return label with highest weight
        best_label = max(label_weights.items(), key=lambda x: x[1])
        print(f"[get_weighted_signboard_decision] Label weights: {label_weights}, Best: {best_label[0]} (weight: {best_label[1]:.2f})")
        return best_label[0]
    
    def calc_areas(self, segmented_image, yolo_output):
        preds = self._safe_get_preds(yolo_output)
        print(f"[calc_areas] Looking for majority_class={self.majority_class}, got {len(preds)} predictions")

        try:
            for pred in preds:
                try:
                    class_id = int(pred[-1])
                except Exception:
                    continue
                if class_id < 0 or class_id >= len(self.class_names):
                    continue
                detected_label = self.class_names[class_id]
                print(f"[calc_areas] Checking pred: class_id={class_id}, label={detected_label}")
                if detected_label == self.majority_class:
                    # Get boxes
                    boxes = pred[:4]

                    # Calculate areas from bounding box
                    try:
                        areas = float(
                            max(0.0, (boxes[2] - boxes[0])) * max(0.0, (boxes[3] - boxes[1])))
                        
                        # Store this detection in signboard history with timestamp
                        current_time = time.time()
                        self.signboard_history.append((detected_label, areas, current_time))
                        
                        # Update max detected area for weighting
                        if areas > self.max_detected_area:
                            self.max_detected_area = areas
                            print(f"[calc_areas] *** MATCH! label={detected_label}, areas={areas:.1f} (NEW MAX), confidence={self.signboard_confidence:.1f}")
                        else:
                            print(f"[calc_areas] *** MATCH! label={detected_label}, areas={areas:.1f}, max={self.max_detected_area:.1f}")
                        
                        # Calculate weighted confidence based on area
                        # Larger areas get exponentially more weight (closer = more confident)
                        area_weight = (areas / 100.0) ** 1.5  # Power scaling for emphasis
                        self.signboard_confidence += area_weight
                        
                        # Store the signboard decision (will be used if boxes disappear)
                        self.stored_signboard = detected_label
                        
                    except Exception:
                        continue

                    self.handle_areas(areas, segmented_image)

                    break

        except Exception:
            print("[calc_areas] Exception during prediction processing")
            pass

    def detect_intersection(self, image, low_height=48, low_window=5, check_height=62, min_width=140, top_thresh=15000):
        # Scan lower band, then verify with upper band
        try:
            print(
                f"[detect_intersection] params: low_height={low_height}, window={low_window}, check_height={check_height}, min_width={min_width}, top_thresh={top_thresh}")
        except Exception:
            pass
        for h in range(low_height, low_height + low_window):
            arr = [x for x, y in enumerate(image[h, :]) if y[0] == 255]
            if len(arr) > 0 and max(arr) - min(arr) > 50:
                # Check upper band
                lineRow = image[min(check_height, image.shape[0]-1), :]
                arr2 = [x for x, y in enumerate(lineRow) if y[0] == 255]
                # dynamic top-center sum
                _, _, sum_top = self._compute_region_sums(image)
                try:
                    low_width = (max(arr) - min(arr)) if len(arr) > 0 else 0
                    up_width = (max(arr2) - min(arr2)) if len(arr2) > 0 else 0
                    print(
                        f"[detect_intersection] candidate at h={h}: low_width={low_width}, up_width={up_width}, sum_top={sum_top}")
                except Exception:
                    pass

                if len(arr2) > 0 and max(arr2) - min(arr2) > min_width and sum_top < top_thresh:
                    print("[detect_intersection] Intersection confirmed!")
                    return True

        print("[detect_intersection] No intersection")
        return False

    def calc_error(self, image):
        """
        Calculates the error between the center of the right lane and the center of the image.
        """

        arr = []
        height = 62
        h_idx = min(height, image.shape[0]-1)
        lineRow = image[h_idx, :]
        for x, y in enumerate(lineRow):
            if y[0] == 255:
                arr.append(x)

        if self.detect_intersection(image):
            print("[calc_error] intersection detected")
            self.intersection_detected = True
            return 0

        if len(arr) > 0:
            center_right_lane = int((min(arr) + max(arr)*2.5)/3.5) - 8
            error = int(image.shape[1]/2) - center_right_lane
            # Base scaling
            error = error * 1.3
            # Stronger response for left-curvy segments
            if error > 0:
                error = int(error * 1.5)
            # If lane is thin (curvy/partial segmentation), amplify left turn to react sooner
            lane_width = max(arr) - min(arr) if len(arr) > 1 else 0
            is_left_context = (error > 0) or (
                getattr(self, 'prev_error', 0) > 0)
            if is_left_context and lane_width < 18:
                error = int(error * 1.4)
            # If very few pixels detected, further boost left bias
            if is_left_context and len(arr) < 10:
                error = int(error * 1.2)
            return error
        else:
            # No lane seen at base row: bias slightly to previous direction
            prev_err = getattr(self, 'prev_error', 0)
            if prev_err > 0:
                return max(10, int(prev_err * 0.6))
            elif prev_err < 0:
                return min(-6, int(prev_err * 0.5))
            return 0

    def PID(self, error, p, i, d):
        """
        Calculates the PID output for the specified error.
        """
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.pre_t
        self.pre_t = time.time()
        if delta_t <= 0:
            delta_t = 1e-3
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D

        # Asymmetric clamp: allow stronger left turns
        max_left = 30
        max_right = 25
        if angle > max_left:
            angle = max_left
        elif angle < -max_right:
            angle = -max_right

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
            speed = 25
        elif 10 <= abs(angle) <= 20:
            speed = 1
        else:
            speed = 1
        return speed

    def verify_intersection(self, image, height=48, window=5):
        for h in range(height, height+window):
            arr = [x for x, y in enumerate(image[h, :]) if y[0] == 255]
            if len(arr) > 0 and max(arr) - min(arr) > 50:
                return True
        return False
