import numpy as np
import time

from utils.utils import find_majority


class Controller():
    """
    Main controller for autonomous car navigation with signboard detection and intersection handling.
    Uses a 4-state machine: COLLECTING → CONFIRMING → WAITING → TURNING
    """
    
    # ==================== HYPERPARAMETERS - TUNE THESE ====================
    
    # Signboard Collection
    SAMPLES_REQUIRED = 20              # Number of samples to collect before confirming signboard
    EARLY_TRIGGER_SAMPLES = 10         # Minimum samples for early trigger if intersection detected
    LEFT_SIGN_BOOST = 3                # Multiplier for 'left' sign detection (boost importance)
    MIN_BBOX_AREA = 200                # Minimum bounding box area to accept detection (filter small/far signs)
    
    # Area Confidence Weighting
    AREA_WEIGHT_DIVISOR = 100.0        # Divisor for area in confidence formula
    AREA_WEIGHT_EXPONENT = 1.5         # Exponent for area weighting: (area/divisor)^exponent
    
    # Turning Sequence Timing
    TURNING_MAX_FRAMES = 25            # Maximum frames for turn execution
    
    # Turn Angles (degrees)
    ANGLE_RIGHT = -25                  # Right turn angle
    ANGLE_LEFT_CASE1 = 25              # Left turn angle (case 1: top_corner > threshold)
    ANGLE_LEFT_CASE2 = 25              # Left turn angle (case 2: default)
    ANGLE_NO_LEFT = -25                # Angle for 'no left' sign (turn right)
    ANGLE_NO_RIGHT = 25                # Angle for 'no right' sign (turn left)
    ANGLE_NO_STRAIGHT_LEFT = 25        # Angle for 'no straight' sign (prefer left)
    ANGLE_NO_STRAIGHT_RIGHT = -24      # Angle for 'no straight' sign (prefer right)
    
    # Corner Sum Thresholds (for turn decision logic)
    TOP_CORNER_HIGH_THRESH = 17_000    # High threshold for top corner sum
    TOP_CORNER_MID_THRESH = 17_500     # Mid threshold for top corner sum
    LEFT_CORNER_THRESH = 2_000         # Threshold for left corner sum
    
    # Intersection Detection Parameters
    INTERSECTION_MIN_WIDTH = 95       # Minimum lane width to detect intersection
    INTERSECTION_LOW_HEIGHT = 50       # Starting height for width check
    INTERSECTION_CHECK_WINDOW = 5      # Window size for consecutive height checks
    INTERSECTION_CHECK_HEIGHT = 62     # Maximum height to check
    # INTERSECTION_TOP_THRESH = 15_000   # Threshold for top region sum
    
    # System Reset
    RESET_COUNTER_LIMIT = 400          # Frames before automatic reset
    
    # Logging (set to False to reduce terminal output)
    VERBOSE_LOGGING = True             # Enable detailed state transition logs
    
    # ==================================================================
    
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

        # Signboard Detection & Storage
        self.stored_class_names = []       # List to store detected labels for majority voting
        self.signboard_history = []        # List of tuples: (label, area, timestamp)
        self.max_detected_area = 0         # Track maximum area detected
        self.stored_signboard = None       # Latest detected signboard
        self.signboard_confidence = 0      # Weighted confidence score
        self.majority_class = ""           # Confirmed majority class after voting
        
        # Turning State
        self.turning_counter = 0           # Counter to track turning frames
        self.angle_turning = 0             # Angle to turn the car

        # Corner Pixel Sums (for turn decision logic)
        self.sum_left_corner = 0
        self.sum_right_corner = 0
        self.sum_top_corner = 0

        # Image Masking Flags
        self.mask_l = False                # Mask left side of image
        self.mask_r = False                # Mask right side of image
        self.mask_lr = False               # Mask both left and right
        self.mask_t = False                # Mask top of image

        # State Machine Flags
        self.next_step = False             # Next step flag for turning
        self.is_turning = False            # Currently executing turn
        self.waiting_for_intersection = False  # Waiting for intersection after signboard confirmed
        self.intersection_detected = False # Intersection detected flag (set by calc_error)

        # Turn Type Flags (set during _apply_signboard_decision)
        self.is_turn_left_case_1 = False   # Left turn case 1 (high top corner)
        self.is_turn_left_case_2 = False   # Left turn case 2 (default)
        self.is_no_turn_right_case_1 = False  # No right case 1 (turn left)
        self.is_no_turn_right_case_2 = False  # No right case 2 (go straight)

        # System Reset Counter
        self.reset_counter = 0

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
        """
        Compute region sums from segmented image.
        NOTE: Top corners often have no road (black), so sums may be 0.
        Instead, check regions LOWER in the image where road is more visible.
        """
        try:
            h, w = img.shape[0], img.shape[1]
        except Exception as e:
            print(f"[ERROR] Invalid image shape for region sums: {e}")
            return 0, 0, 0

        # Check image shape and content
        if len(img.shape) < 3:
            print(f"[ERROR] Image has wrong dimensions: {img.shape}, expected 3 channels")
            return 0, 0, 0

        # Instead of top corners (which are often black), check LOWER regions where road is visible
        # Use middle section of image where road is typically segmented
        
        # For 80x160 image: check around row 40-60 (middle to lower area)
        check_row_start = max(0, int(h * 0.5))  # Start at 50% height
        check_row_end = min(h, int(h * 0.75))    # End at 75% height
        
        # Left side (left 30% of width)
        left_col_end = int(w * 0.3)
        sum_left = np.sum(img[check_row_start:check_row_end, :left_col_end, 0])
        
        # Right side (right 30% of width)  
        right_col_start = int(w * 0.7)
        sum_right = np.sum(img[check_row_start:check_row_end, right_col_start:, 0])
        
        # Center (middle 40% of width)
        center_col_start = int(w * 0.3)
        center_col_end = int(w * 0.7)
        sum_top = np.sum(img[check_row_start:check_row_end, center_col_start:center_col_end, 0])
        
        # Debug logging
        if self.waiting_for_intersection or self.intersection_detected:
            print(f"[_compute_region_sums] Image: {h}x{w}, checking rows {check_row_start}-{check_row_end}")
            print(f"[_compute_region_sums] Left (cols 0-{left_col_end}): {int(sum_left)}")
            print(f"[_compute_region_sums] Right (cols {right_col_start}-{w}): {int(sum_right)}")
            print(f"[_compute_region_sums] Center (cols {center_col_start}-{center_col_end}): {int(sum_top)}")

        return int(sum_left), int(sum_right), int(sum_top)

    def reset(self):
        """Reset all state variables to default values after turn completion or timeout."""
        # Signboard Detection
        self.stored_class_names = []
        self.signboard_history = []
        self.max_detected_area = 0
        self.stored_signboard = None
        self.signboard_confidence = 0
        self.majority_class = ""
        
        # State Machine
        self.waiting_for_intersection = False
        self.intersection_detected = False
        self.is_turning = False
        self.next_step = False
        
        # Turning State
        self.turning_counter = 0
        self.angle_turning = 0
        
        # Turn Type Flags
        self.is_turn_left_case_1 = False
        self.is_turn_left_case_2 = False
        self.is_no_turn_right_case_1 = False
        self.is_no_turn_right_case_2 = False
        
        # Image Masking
        self.mask_l = False
        self.mask_r = False
        self.mask_lr = False
        self.mask_t = False
        
        # Reset counter
        self.reset_counter = 0
        
        # Lane tracking
        self.last_center_right_lane = None
        self.lost_lane_frames = 0
        self.prev_error = 0.0

    def control(self, segmented_image, yolo_output):
        """
        Main control loop implementing the 4-state machine:
        1. TURNING - Execute turn sequence
        2. WAITING - Monitor for intersection after signboard confirmed
        3. COLLECTING - Gather signboard samples
        4. CONFIRMING - Vote on majority and enter waiting state
        """
        # Safety reset after many frames to avoid stale state
        if self.reset_counter >= self.RESET_COUNTER_LIMIT:
            print(f"[control] 🔄 Auto-reset after {self.reset_counter} frames")
            self.reset()

        # Calculate area of left, right, and top corner of the segmented image
        self.sum_left_corner, self.sum_right_corner, self.sum_top_corner = self._compute_region_sums(
            segmented_image)

        # ==================== 4-STATE MACHINE ====================
        
        # State 1: TURNING - Execute turn sequence
        if self.is_turning:
            self.handle_turning()
        
        # State 2: WAITING - Monitor for intersection after signboard confirmed
        elif self.waiting_for_intersection:
            if self.VERBOSE_LOGGING:
                print(f"[control] 🔍 Waiting for intersection... (signboard: {self.majority_class})")
            
            # Check if intersection detected (by calc_error OR current frame)
            if self.intersection_detected or self.detect_intersection(segmented_image):
                if self.VERBOSE_LOGGING:
                    print(f"[control] ✓ INTERSECTION REACHED! Applying signboard: {self.majority_class}")
                self.intersection_detected = True
                self.waiting_for_intersection = False
                
                # Apply the stored signboard decision
                self._apply_signboard_decision()
        
        # State 3: COLLECTING - Gather signboard samples (< SAMPLES_REQUIRED)
        elif len(self.stored_class_names) < self.SAMPLES_REQUIRED:
            preds = self._safe_get_preds(yolo_output)
            
            for pred in preds:
                try:
                    class_id = int(pred[-1])
                except Exception:
                    continue
                if class_id < 0 or class_id >= len(self.class_names):
                    continue
                
                # Calculate bounding box area and skip if too small
                boxes = pred[:4]
                try:
                    bbox_area = float(max(0.0, (boxes[2] - boxes[0])) * max(0.0, (boxes[3] - boxes[1])))
                    if bbox_area < self.MIN_BBOX_AREA:
                        if self.VERBOSE_LOGGING:
                            print(f"[control] ⏭️  Skipping detection with small area: {bbox_area:.1f} < {self.MIN_BBOX_AREA}")
                        continue
                except Exception:
                    continue
                
                label = self.class_names[class_id]
                if label in self.traffic_lights:
                    self.stored_class_names.append(label)
                    if self.VERBOSE_LOGGING:
                        print(f"[control] 📝 Added sign '{label}' to stored list (count={len(self.stored_class_names)})")
                    
                    # Store in signboard history with area for weighting
                    try:
                        areas = bbox_area  # Reuse the calculated area
                        current_time = time.time()
                        self.signboard_history.append((label, areas, current_time))
                        if areas > self.max_detected_area:
                            self.max_detected_area = areas
                        # Apply area weighting formula from hyperparameters
                        area_weight = (areas / self.AREA_WEIGHT_DIVISOR) ** self.AREA_WEIGHT_EXPONENT
                        self.signboard_confidence += area_weight
                        self.stored_signboard = label
                        if self.VERBOSE_LOGGING:
                            print(f"[control] 💾 Stored: {label}, area={areas:.1f}, confidence={self.signboard_confidence:.1f}")
                    except Exception:
                        pass

        # State 4: CONFIRMING - Samples collected, determine majority and enter waiting
        elif len(self.stored_class_names) >= self.SAMPLES_REQUIRED:
            # Determine the majority class from collected samples
            self.majority_class = find_majority(self.stored_class_names)[0]
            if self.VERBOSE_LOGGING:
                print(f"[control] ✅ Signboard confirmed: {self.majority_class} (from {len(self.stored_class_names)} samples)")
                print("[control] 🔍 Now waiting for intersection...")
            
            # Enter intersection waiting state
            self.waiting_for_intersection = True

        # Increment frame counter and return
        self.reset_counter += 1
        return self.sendBack_angle, self.sendBack_speed, self.next_step, self.mask_l, self.mask_r

    def _apply_signboard_decision(self):
        """
        Apply the stored majority_class signboard decision by setting turn parameters.
        This method is called when intersection is reached after signboard is confirmed.
        Uses hyperparameters for angles and thresholds.
        """
        if self.VERBOSE_LOGGING:
            print(f"[_apply_signboard_decision] Applying: {self.majority_class}")
        
        if self.majority_class == 'right':
            self.angle_turning = self.ANGLE_RIGHT
            self.is_turning = True
            if self.VERBOSE_LOGGING:
                print(f"[_apply_signboard_decision] → RIGHT TURN ({self.ANGLE_RIGHT}°)")
            
        elif self.majority_class == 'left':
            if self.sum_top_corner > self.TOP_CORNER_HIGH_THRESH:
                self.angle_turning = self.ANGLE_LEFT_CASE1
                self.is_turn_left_case_1 = True
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → LEFT TURN Case 1 ({self.ANGLE_LEFT_CASE1}°)")
            else:
                self.angle_turning = self.ANGLE_LEFT_CASE2
                self.is_turn_left_case_2 = True
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → LEFT TURN Case 2 ({self.ANGLE_LEFT_CASE2}°)")
            self.is_turning = True
            
        elif self.majority_class == 'straight':
            self.angle_turning = 0
            self.is_turning = True
            self.mask_l = True
            self.mask_r = True
            if self.VERBOSE_LOGGING:
                print("[_apply_signboard_decision] → STRAIGHT (0°, mask L+R)")
            
        elif self.majority_class == 'no right':
            print(f"[DEBUG NO RIGHT] sum_left: {self.sum_left_corner}, sum_top: {self.sum_top_corner}")
            print(f"[DEBUG NO RIGHT] LEFT_CORNER_THRESH: {self.LEFT_CORNER_THRESH}, TOP_CORNER_MID_THRESH: {self.TOP_CORNER_MID_THRESH}")
            
            # Check if corner sums are valid (not all zeros)
            if self.sum_left_corner == 0 and self.sum_top_corner == 0 and self.sum_right_corner == 0:
                # Fallback: No road detected in corners, default to LEFT for "no right"
                print("[DEBUG NO RIGHT] ⚠️ All corner sums are 0! Using fallback: TURN LEFT")
                self.angle_turning = self.ANGLE_NO_RIGHT  # Turn left
                self.is_no_turn_right_case_1 = True
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO RIGHT (FALLBACK): Turn left ({self.ANGLE_NO_RIGHT}°)")
            elif self.sum_left_corner > self.LEFT_CORNER_THRESH and self.sum_top_corner < self.TOP_CORNER_MID_THRESH:
                # Normal case: left corner has road, top doesn't → turn left
                self.angle_turning = self.ANGLE_NO_RIGHT
                self.is_no_turn_right_case_1 = True
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO RIGHT: Turn left ({self.ANGLE_NO_RIGHT}°)")
            else:
                # Go straight
                self.angle_turning = 0
                self.mask_l = True
                self.mask_r = True
                self.is_no_turn_right_case_2 = True
                if self.VERBOSE_LOGGING:
                    print("[_apply_signboard_decision] → NO RIGHT: Go straight (mask L+R)")
            self.is_turning = True
            
        elif self.majority_class == 'stop':
            self.angle_turning = 0
            self.is_turning = True
            if self.VERBOSE_LOGGING:
                print("[_apply_signboard_decision] → STOP (0°)")
            
        elif self.majority_class == 'no left':
            print(f"[DEBUG NO LEFT] sum_right: {self.sum_right_corner}, sum_top: {self.sum_top_corner}, sum_left: {self.sum_left_corner}")
            
            # For "no left" sign: Default is to turn RIGHT (since we can't go left)
            # Only go straight if there's significantly more road ahead than on the right
            if self.sum_top_corner > self.sum_right_corner * 2:
                # Lots of road ahead, minimal on right → go straight
                self.angle_turning = 0
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO LEFT: Go straight (0°) [top={self.sum_top_corner} > right*2={self.sum_right_corner*2}]")
            else:
                # Right has decent road or not much straight ahead → turn right
                self.angle_turning = self.ANGLE_NO_LEFT
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO LEFT: Turn right ({self.ANGLE_NO_LEFT}°) [top={self.sum_top_corner} <= right*2={self.sum_right_corner*2}]")
            self.is_turning = True
            
        elif self.majority_class == 'no_straight':
            if self.sum_left_corner > self.sum_right_corner * 4:
                self.angle_turning = self.ANGLE_NO_STRAIGHT_LEFT
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO STRAIGHT: Turn left ({self.ANGLE_NO_STRAIGHT_LEFT}°)")
            else:
                self.angle_turning = self.ANGLE_NO_STRAIGHT_RIGHT
                if self.VERBOSE_LOGGING:
                    print(f"[_apply_signboard_decision] → NO STRAIGHT: Turn right ({self.ANGLE_NO_STRAIGHT_RIGHT}°)")
            self.is_turning = True
            
        else:
            if self.VERBOSE_LOGGING:
                print(f"[_apply_signboard_decision] ⚠️ Unhandled class: {self.majority_class}")

    def handle_turning(self):
        """Execute the turn sequence based on turn type and counter."""
        speed = 0
        angle = 0

        # Check turning counter against hyperparameter
        if self.turning_counter < self.TURNING_MAX_FRAMES:
            if self.majority_class == 'left':
                if self.is_turn_left_case_1:
                    speed = 0
                    if self.turning_counter <= 3:  # Hard
                        angle = 25
                    elif self.turning_counter > 3 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
                elif self.is_turn_left_case_2:
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 25
                    elif self.turning_counter > 1 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.majority_class == 'right':
                speed = 0
                # if self.turning_counter <= 1:
                #     angle = -5
                if self.turning_counter >= 0 and self.turning_counter < 7:
                    angle = self.angle_turning
                else:
                    self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.majority_class == 'no left':
                speed = 0
                if self.turning_counter <= 1:
                    angle = 2
                elif self.turning_counter > 1 and self.turning_counter <= 5:
                    angle = -25
                else:
                    self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.majority_class == 'no right':
                if self.is_no_turn_right_case_1:    # Left hard
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 5
                    elif self.turning_counter > 1 and self.turning_counter <= 7:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
                elif self.is_no_turn_right_case_2:  # Left
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
                elif self.is_no_turn_right_case_3:  # Straight (left of map)
                    speed = 0
                    if self.turning_counter <= 9:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
                else:   # Straight: self.is_no_turn_right_case_4
                    speed = 0
                    if self.turning_counter <= 9:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.majority_class == 'straight':
                speed = 0
                if self.turning_counter <= 7:
                    angle = self.angle_turning
                else:
                    self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.majority_class == 'no_straight':
                if self.is_turn_left:  # Left
                    speed = 0
                    if self.turning_counter <= 11:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 5:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
                else:  # Right
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 5:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = self.TURNING_MAX_FRAMES
            elif self.majority_class == 'stop':

                if self.turning_counter <= 3:
                    speed = -25
                    angle = 0
                elif self.turning_counter > 3 and self.turning_counter <= 15:
                    speed = 0
                    angle = self.angle_turning
                else:
                    self.turning_counter = self.TURNING_MAX_FRAMES

            elif self.angle_turning == 20:  # Forced left turn at intersection
                speed = 20  # Higher speed to pass quickly
                if self.turning_counter <= 30:  # Hold for longer
                    angle = self.angle_turning
                else:
                    self.turning_counter = self.TURNING_MAX_FRAMES

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

        elif self.turning_counter >= self.TURNING_MAX_FRAMES:
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

    def detect_intersection(self, image, 
                          low_height=None, low_window=None, check_height=None, 
                          min_width=None, top_thresh=None):
        """
        Simplified intersection detection: Check if road is wide at a specific height.
        Intersection = Wide road (> threshold) + Road visible ahead (header check)
        """
        # Use hyperparameters if not specified
        low_height = low_height if low_height is not None else self.INTERSECTION_LOW_HEIGHT
        min_width = min_width if min_width is not None else self.INTERSECTION_MIN_WIDTH
        check_height = check_height if check_height is not None else self.INTERSECTION_CHECK_HEIGHT
        
        if self.VERBOSE_LOGGING:
            print(f"[detect_intersection] Checking at height={low_height}, min_width={min_width}, header_height={check_height}")
        
        # Check road width at the specified height (near bottom)
        h_idx = min(low_height, image.shape[0] - 1)
        lineRow = image[h_idx, :]
        road_pixels = [x for x, y in enumerate(lineRow) if y[0] == 255]
        
        if len(road_pixels) == 0:
            if self.VERBOSE_LOGGING:
                print("[detect_intersection] No road detected at check height")
            return False
        
        road_width = max(road_pixels) - min(road_pixels)
        
        # Check for road ahead (header) - ensures we're approaching intersection, not just wide lane
        header_row = image[min(check_height, image.shape[0] - 1), :]
        header_pixels = [x for x, y in enumerate(header_row) if y[0] == 255]
        has_header = len(header_pixels) > 0
        
        if self.VERBOSE_LOGGING:
            print(f"[detect_intersection] road_width={road_width}, has_header={has_header} (header_pixels={len(header_pixels)})")
        
        # Simple logic: Wide road + Road visible ahead = Intersection
        is_intersection = road_width > min_width and has_header
        
        if is_intersection:
            if self.VERBOSE_LOGGING:
                print(f"[detect_intersection] ✅ INTERSECTION! width={road_width} > {min_width}, header=yes")
            return True
        else:
            if self.VERBOSE_LOGGING:
                if road_width <= min_width:
                    print(f"[detect_intersection] ❌ Road too narrow: {road_width} <= {min_width}")
                else:
                    print(f"[detect_intersection] ❌ No header (road ahead not visible)")
            return False

    def calc_error(self, image):
        """
        Calculates the error between the center of the right lane and the center of the image.
        Only checks for intersections if a signboard has been detected (waiting_for_intersection=True).
        """

        arr = []
        height = 62
        h_idx = min(height, image.shape[0]-1)
        lineRow = image[h_idx, :]
        for x, y in enumerate(lineRow):
            if y[0] == 255:
                arr.append(x)

        # Only check for intersection if we're waiting for one (signboard detected)
        if self.waiting_for_intersection and self.detect_intersection(image):
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
