import numpy as np
import time
from typing import Optional

from utils.utils import find_majority


class Controller:
    """
    Lane-keeping controller with optional detection-assisted intersection handling.

    Key features:
    - Tunable parameters grouped under self.params for easy adjustment
    - Toggle `use_detection_model` to turn YOLO-based decisioning on/off
    - When detection is OFF, a simple intersection detector uses the segmentation mask
      and instructs the car to go straight through intersections
    """

    def __init__(self, *, use_detection_model: bool = True, debug: bool = False, **param_overrides):
        # Global toggles
        self.use_detection_model = use_detection_model
        self.debug = debug

        # Initialize variables for PID control and traffic signs detection
        self.error_arr = np.zeros(5)  # Will be resized after params are set
        self.error_sp = np.zeros(5)

        self.pre_t = time.time()      # Store the current time for steering control
        self.pre_t_spd = time.time()  # Store the current time for speed control

        self.sendBack_angle = 0       # Initialize the steering angle to 0
        self.sendBack_speed = 0       # Initialize the speed to 0

        # List of all the possible traffic light labels (from detector)
        self.traffic_lights = ['no_turn_right', 'stop', 'straight', 'turn_left', 'turn_right']

        # List of all the possible object detection labels
        self.class_names = ['no_turn_right', 'stop', 'straight', 'turn_left', 'turn_right']

        # Buffer for majority vote
        self.stored_class_names = []

        # Runtime state
        self.majority_class = ""
        self.start_cal_area = False
        self.turning_counter = 0
        self.angle_turning = 0

        # Cached segmentation region sums
        self.sum_left_corner = 0
        self.sum_right_corner = 0
        self.sum_top_corner = 0

        # Optional masks for visualization/processing
        self.mask_l = False
        self.mask_r = False
        self.mask_lr = False
        self.mask_t = False

        self.next_step = False
        self.is_turning = False

        self.reset_counter = 0

        # Turning case flags
        self.is_turn_left = False
        self.is_turn_right = False
        self.is_straight = False

        self.is_no_turn_right_case_1 = False
        self.is_no_turn_right_case_2 = False
        self.is_no_turn_right_case_3 = False
        self.is_no_turn_right_case_4 = False

        self.is_turn_left_case_1 = False
        self.is_turn_left_case_2 = False

        # Tunable parameters (edit here or pass overrides via kwargs)
        self.params = {
            # Majority vote settings
            'majority_buffer_size': 20,  # number of detections to collect before deciding

            # Turning behavior
            'turn_max_counter': 25,      # number of frames to apply turning behavior
            'default_turn_speed': 30,    # speed applied during turning when not otherwise set

            # PID control
            'pid_window': 5,             # number of past errors to keep for I/D
            'pid_max_abs_angle': 25,     # clamp for resulting angle
            'pid_dt_epsilon': 1e-6,      # minimum delta time for D/I stability

            # Segmentation corner windows (assumes resized to 160x80)
            'right_corner_y': 50,        # height of top region for right corner
            'right_corner_x': 50,        # width of right corner window
            'left_corner_y': 24,         # height of top region for left corner
            'left_corner_x': 24,         # width of left corner window
            'top_center_y': 24,          # height of top center band
            'top_center_x0': 134,        # start x of top center band (used if top_band_mode='param')
            'top_center_x1': 184,        # end x of top center band (used if top_band_mode='param')
            # How to define the top band horizontally
            'top_band_mode': 'center',   # 'center' uses fractions below; 'param' uses x0/x1 above
            'top_center_frac_left': 0.35,  # left fraction (0..1) from image width
            'top_center_frac_right': 0.65, # right fraction (0..1) from image width
            # Where to place the top band vertically: 'bottom' (near corners) or 'mid'
            'top_vertical_mode': 'bottom',

            # calc_error parameters
            'error_row_y': 48,           # row index to sample for lane detection
            'error_lane_value': 255,     # pixel value that indicates lane presence in channel 0
            'error_center_weight_min': 1.0,  # weight for min(arr)
            'error_center_weight_max': 2.5,  # weight for max(arr)
            'error_center_div': 3.5,     # divisor for weighted center
            'error_center_offset': -5,   # offset applied to computed center
            'error_gain': 1.3,           # scale factor applied to computed error

            # Detection-less intersection heuristic (segmentation based)
            'intersect_min_left_sum': 2000,
            'intersect_min_right_sum': 2000,
            'intersect_min_top_sum': 15000,
            # Count-based thresholds (robust when mask values are 0/1 or colored in other channels)
            'intersect_min_left_count': 50,
            'intersect_min_right_count': 50,
            'intersect_min_top_count': 50,
            # Ratio-based thresholds (count / window_area)
            'intersect_min_left_ratio': 0.05,
            'intersect_min_right_ratio': 0.05,
            'intersect_min_top_ratio': 0.02,
            # Require top band in decision (set True to avoid sharp-turn false positives)
            'intersect_require_top': True,
            # If top is not required, how strong the corners should be by ratio
            'intersect_corners_only_min_ratio': 0.12,
            # Additional constraints to reduce FPs
            'intersect_lr_min_balance': 0.5,   # min(left/right) / max(left/right) using ratios
            'intersect_min_streak': 3,         # require N consecutive frames

            # Optional speed curve based on angle (used by calc_speed)
            'speed_angle_low': 10,
            'speed_angle_mid': 20,
            'speed_when_low': 25,
            'speed_when_mid': 1,
            'speed_when_high': 1,
        }
        # Apply user overrides
        self.params.update(param_overrides or {})

        # Resize error buffers based on params
        _win = max(1, int(self.params.get('pid_window', 5)))
        self.error_arr = np.zeros(_win)
        self.error_sp = np.zeros(_win)
        # Intersection decision smoothing
        self.intersect_streak = 0

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
        self.is_turn_left_case_2 = False
        self.intersect_streak = 0

    def control(self, segmented_image, yolo_output: Optional[object]):
        # if self.majority_class == "turn_left":
        #     self.reset_counter += 1
        if self.reset_counter >= 50:
            self.reset()
            print("Reset" * 200)

        # Calculate stats of left, right, and mid-top regions of the segmented image
        # We compute both sum on channel 0 (legacy) and non-zero counts across channels
        H, W = segmented_image.shape[:2]
        rc_y = int(self.params['right_corner_y'])
        rc_x = int(self.params['right_corner_x'])
        lc_y = int(self.params['left_corner_y'])
        lc_x = int(self.params['left_corner_x'])
        tc_y = int(self.params['top_center_y'])
        # Compute top band x-range based on mode
        top_mode = self.params.get('top_band_mode', 'center')
        if top_mode == 'center':
            fl = float(self.params.get('top_center_frac_left', 0.35))
            fr = float(self.params.get('top_center_frac_right', 0.65))
            tc_x0 = int(max(0, min(W, round(W * fl))))
            tc_x1 = int(max(tc_x0, min(W, round(W * fr))))
        else:
            tc_x0 = int(self.params.get('top_center_x0', max(0, int(W * 0.3))))
            tc_x1 = int(self.params.get('top_center_x1', min(W, int(W * 0.7))))

        rc_y = max(0, min(rc_y, H))
        rc_x = max(0, min(rc_x, W))
        lc_y = max(0, min(lc_y, H))
        lc_x = max(0, min(lc_x, W))
        tc_y = max(0, min(tc_y, H))
        tc_x0 = max(0, min(tc_x0, W))
        tc_x1 = max(tc_x0, min(tc_x1, W))

        # Regions: bottom corners and a mid vertical band near center for top check
        y0_right, y1_right = max(0, H - rc_y), H
        y0_left, y1_left = max(0, H - lc_y), H
        top_v_mode = self.params.get('top_vertical_mode', 'bottom')
        if top_v_mode == 'bottom':
            mb_y0 = max(0, H - tc_y)
            mb_y1 = H
        else:
            mb_y0 = max(0, int(H * 0.35))
            mb_y1 = min(H, mb_y0 + tc_y)

        reg_right = segmented_image[y0_right:y1_right, W - rc_x if rc_x <= W else 0: W]
        reg_left = segmented_image[y0_left:y1_left, :lc_x]
        reg_top = segmented_image[mb_y0:mb_y1, tc_x0:tc_x1]

        # Window areas for ratio checks
        area_r = max(1, (y1_right - y0_right) * min(W, rc_x))
        area_l = max(1, (y1_left - y0_left) * min(W, lc_x))
        area_t = max(1, (mb_y1 - mb_y0) * max(0, tc_x1 - tc_x0))

        def region_stats(region):
            # Sum on channel 0 (if present), and non-zero across all channels
            if region.ndim == 3 and region.shape[2] > 0:
                ch0 = region[:, :, 0]
                any_mask = (region.max(axis=2) > 0)
            else:
                ch0 = region
                any_mask = (region > 0)
            pixel_sum = int(ch0.sum())
            count_nonzero = int(np.count_nonzero(any_mask))
            return pixel_sum, count_nonzero

        sum_r, cnt_r = region_stats(reg_right)
        sum_l, cnt_l = region_stats(reg_left)
        sum_t, cnt_t = region_stats(reg_top)

        # Preserve legacy sums for detection-based turning logic
        self.sum_right_corner = sum_r
        self.sum_left_corner = sum_l
        self.sum_top_corner = sum_t

        # New: store counts and ratios for robust intersection detection
        self.count_right_corner = cnt_r
        self.count_left_corner = cnt_l
        self.count_top_corner = cnt_t
        self.ratio_right_corner = cnt_r / float(area_r)
        self.ratio_left_corner = cnt_l / float(area_l)
        self.ratio_top_corner = cnt_t / float(area_t)

        if self.debug:
            any_mask_all = (segmented_image.max(axis=2) > 0) if segmented_image.ndim == 3 else (segmented_image > 0)
            total_nonzero = int(np.count_nonzero(any_mask_all))
            print("Is calculate areas:", self.start_cal_area)
            print("Is turning:", self.is_turning)
            print(f"[MASK NONZERO] total:{total_nonzero} of {H*W}")
            print(f"[WINDOW SUMS] L:{self.sum_left_corner} R:{self.sum_right_corner} T:{self.sum_top_corner}")
            print(f"[WINDOW CNTS] L:{self.count_left_corner} R:{self.count_right_corner} T:{self.count_top_corner}")
            print(f"[WINDOW RATIOS] L:{self.ratio_left_corner:.3f} R:{self.ratio_right_corner:.3f} T:{self.ratio_top_corner:.3f}")
            print(f"[WINDOWS] L[y:{y0_left}-{y1_left}, x:0-{lc_x}] R[y:{y0_right}-{y1_right}, x:{W-rc_x}-{W}] T[y:{mb_y0}-{mb_y1}, x:{tc_x0}-{tc_x1}] hmode:{top_mode} vmode:{top_v_mode}")

        # 1) Continue turning if already in that phase
        if self.start_cal_area:
            self.calc_areas(segmented_image, yolo_output)

        elif self.is_turning:
            self.handle_turning()

        # 2) Detection-assisted mode
        elif self.use_detection_model and (yolo_output is not None):
            if len(self.stored_class_names) < self.params['majority_buffer_size']:
                preds = yolo_output.boxes.data.detach().cpu().numpy()
                if self.debug:
                    print("Preds:", preds)
                    for pred in preds:
                        print(f"Bounding Box: {pred[:4]}, Confidence: {pred[4]}, Class ID: {pred[5]}")

                for pred in preds:
                    class_id = int(pred[-1])
                    if 0 <= class_id < len(self.class_names):
                        label = self.class_names[class_id]
                        if label in self.traffic_lights:
                            if label == 'turn_left':
                                self.stored_class_names.extend(['turn_left'] * 3)
                            else:
                                self.stored_class_names.append(label)
            else:
                self.majority_class = find_majority(self.stored_class_names)[0]
                self.start_cal_area = True

        # 3) Segmentation-only mode
        else:
            if self.detect_intersection():
                self.majority_class = 'straight'
                self.angle_turning = 0
                self.is_turning = True
                self.start_cal_area = False
                self.handle_turning()

        return self.sendBack_angle, self.sendBack_speed, self.next_step, self.mask_l, self.mask_r

    def detect_intersection(self) -> bool:
        """
        Decide if approaching an intersection based on segmentation windows.
        Consider it an intersection if thresholds are met: primarily uses count thresholds,
        with optional top-band requirement and ratio checks; falls back to sums if counts are zero.
        """
        require_top = bool(self.params.get('intersect_require_top', False))

        # Prefer count-based thresholds for robustness across different mask encodings
        l_cnt_ok = self.count_left_corner >= int(self.params['intersect_min_left_count'])
        r_cnt_ok = self.count_right_corner >= int(self.params['intersect_min_right_count'])
        t_cnt_ok = self.count_top_corner >= int(self.params['intersect_min_top_count'])

        # Ratio-based checks
        l_ratio_ok = self.ratio_left_corner >= float(self.params['intersect_min_left_ratio'])
        r_ratio_ok = self.ratio_right_corner >= float(self.params['intersect_min_right_ratio'])
        t_ratio_ok = self.ratio_top_corner >= float(self.params['intersect_min_top_ratio'])

        # Left/Right balance to avoid one-sided sharp-turn false positives
        max_lr = max(self.ratio_left_corner, self.ratio_right_corner, 1e-6)
        min_lr = min(self.ratio_left_corner, self.ratio_right_corner)
        lr_balance = min_lr / max_lr

        decision_counts = (l_cnt_ok and r_cnt_ok and (t_cnt_ok if require_top else True))
        decision_ratios = (l_ratio_ok and r_ratio_ok and (t_ratio_ok if require_top else True))
        decision = (decision_counts or decision_ratios) and (lr_balance >= float(self.params['intersect_lr_min_balance']))

        # If top is not required, allow a corners-only decision using a stronger ratio threshold
        if not require_top and not decision:
            corners_only_min = float(self.params['intersect_corners_only_min_ratio'])
            # Also enforce left/right balance even in corners-only mode
            decision = (
                self.ratio_left_corner >= corners_only_min and
                self.ratio_right_corner >= corners_only_min and
                lr_balance >= float(self.params['intersect_lr_min_balance'])
            )

        # Fallback to pixel-value sums if counts are all zero (unexpected but possible)
        if (self.count_left_corner == 0 and self.count_right_corner == 0 and self.count_top_corner == 0):
            l_ok = self.sum_left_corner >= self.params['intersect_min_left_sum']
            r_ok = self.sum_right_corner >= self.params['intersect_min_right_sum']
            t_ok = self.sum_top_corner >= self.params['intersect_min_top_sum']
            decision = l_ok and r_ok and (t_ok if require_top else True)

        # Apply streak requirement to stabilize decision over frames
        min_streak = int(self.params.get('intersect_min_streak', 1))
        if decision:
            self.intersect_streak += 1
        else:
            self.intersect_streak = 0
        final_decision = self.intersect_streak >= max(1, min_streak)

        if self.debug:
            print(
                f"[INTERSECT CHECK] CNTS(L/R/T): {self.count_left_corner}/{self.count_right_corner}/{self.count_top_corner} "
                f"TH: {self.params['intersect_min_left_count']}/{self.params['intersect_min_right_count']}/{self.params['intersect_min_top_count']} "
                f"RAT(L/R/T): {self.ratio_left_corner:.3f}/{self.ratio_right_corner:.3f}/{self.ratio_top_corner:.3f} "
                f"TH: {self.params['intersect_min_left_ratio']}/{self.params['intersect_min_right_ratio']}/{self.params['intersect_min_top_ratio']} "
                f"require_top={require_top} balance={lr_balance:.2f} streak={self.intersect_streak}/{min_streak} -> {final_decision}"
            )

        return bool(final_decision)

    def handle_turning(self):
        if self.debug:
            print("Handle Turning")
        # Default config
        speed = 0
        angle = 0

        # Check turning counter
        MAX_COUNTER = int(self.params['turn_max_counter'])
        if self.turning_counter < MAX_COUNTER:
            if self.debug:
                print('Turning Counter:', self.turning_counter)

            if self.majority_class == 'turn_left':
                if self.is_turn_left_case_1:
                    if self.debug:
                        print("Left hard")
                    speed = 0
                    if self.turning_counter <= 3: # Hard
                        angle = 5
                    elif self.turning_counter > 3 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                elif self.is_turn_left_case_2:
                    if self.debug:
                        print("Left")
                    speed = 0
                    if self.turning_counter <= 1:
                        angle = 5
                    elif self.turning_counter > 1 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'turn_right':
                speed = 0
                if self.turning_counter <= 2:
                    angle = -5
                elif self.turning_counter > 2 and self.turning_counter < 7:
                    angle = self.angle_turning
                else:
                    self.turning_counter = MAX_COUNTER

            elif self.majority_class == 'no_turn_left':
                speed = 0
                if self.turning_counter <= 1:
                    angle = 2
                elif self.turning_counter > 1 and self.turning_counter <= 5:
                    angle = -25
                else:
                    self.turning_counter = MAX_COUNTER
                    
            elif self.majority_class == 'no_turn_right':
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
                elif self.is_no_turn_right_case_3: # Straight (left of map)
                    speed = 0
                    if self.debug:
                        print("Straight")
                    if self.turning_counter <= 2:
                        angle = 0
                    elif self.turning_counter > 2 and self.turning_counter <= 6:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                else:   # Straight: self.is_no_turn_right_case_4
                    speed = 0
                    if self.debug:
                        print("else")
                    if self.turning_counter <= 1:
                        angle = 5
                    elif self.turning_counter > 1 and self.turning_counter <= 8:
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
                if self.is_turn_left: # Left
                    speed = 0
                    if self.turning_counter <= 11:
                        angle = 0
                    elif self.turning_counter > 1 and self.turning_counter <= 5:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER
                else: # Right
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
                            
            # Set default speed
            if speed == 0:
                speed = int(self.params['default_turn_speed'])

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
        if self.debug:
            print("Handle Areas:", areas)

        # ============ Test
        # self.sum_left_corner = np.sum(segmented_image[:12, :12, 0])
        # self.sum_top_corner = np.sum(segmented_image[:12, 67:92, 0])
        # ============

        if self.debug:
            print("self.sum_left_corner", self.sum_left_corner)
            print("self.sum_top_corner", self.sum_top_corner)
            print("self.sum_right_corner", self.sum_right_corner)
        if areas < 100 :
            self.reset() 
        # if areas > 600.0 and self.majority_class == 'turn_right':
        if areas > 615.0 and self.majority_class == 'turn_right':
            # Set angle and error turning
            self.angle_turning = -25
            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas > 550.0 and self.majority_class == 'turn_left':
            if self.sum_top_corner  > 17_000:
                self.is_turn_left_case_1 = True
            else:
                self.is_turn_left_case_2 = True

            self.angle_turning = 25

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas >= 400.0 and self.majority_class == 'no_turn_left':
            if self.sum_right_corner > self.sum_top_corner/2:  # Turn right3
                angle = -25
            else:
                angle = 0  # Straight

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            # Set global angle
            self.angle_turning = angle

        if areas >= 650.0 and self.majority_class == 'no_turn_right':
            if (self.sum_left_corner > 2_000 and self.sum_top_corner < 17_500):# \
                #    or (self.sum_left_corner < 9_000 and self.sum_top_corner < 9_000):
                
                if self.sum_top_corner < 3_000: # Hard
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

        if areas > 580.0 and self.majority_class == 'straight':
            # Set global angle
            self.angle_turning = 0

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            self.mask_l = True
            self.mask_r = True

        if areas > 400.0 and self.majority_class == 'no_straight':
            if self.sum_left_corner > self.sum_right_corner*4:
                # Turn left
                angle = 22
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
        if areas > 380.0 and self.majority_class == 'stop':
            # Set angle and error turning
            self.angle_turning = 0

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

    def calc_areas(self, segmented_image, yolo_output: Optional[object]):
        # Testing
        if self.debug:
            print("Calculating areas!")
            print("Majority class:", self.majority_class)

        # Guard: if no yolo_output, nothing to calculate
        if (yolo_output is None) or (not self.use_detection_model):
            return

        preds = yolo_output.boxes.data.detach().cpu().numpy()

        try:
            for pred in preds:
                class_id = int(pred[-1])
                if self.class_names[class_id] == self.majority_class:
                    # Get boxes
                    boxes = pred[:4]

                    # Calculate areas from bouding boxe
                    areas = (boxes[2] - boxes[0]) * \
                        (boxes[3] - boxes[1])

                    #print("area = ", areas)
                    self.handle_areas(areas, segmented_image)

                    if self.debug:
                        print("self.start_cal_area:", self.start_cal_area)

                    break

        except Exception as e:
            if self.debug:
                print(e)
            pass
    def calc_error(self, image):
        """
        Calculates the error between the center of the right lane and the center of the image.

        Args:
        image: A NumPy array representing the image.

        Returns:
        The error between the center of the right lane and the center of the image.
        """

        arr = []
        # Select a row to scan from the segmentation mask
        height = int(self.params['error_row_y'])
        height = max(0, min(height, image.shape[0] - 1))
        lineRow = image[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == self.params['error_lane_value']:
                arr.append(x)
        if len(arr) > 0:
            w_min = float(self.params['error_center_weight_min'])
            w_max = float(self.params['error_center_weight_max'])
            denom = float(self.params['error_center_div'])
            offset = float(self.params['error_center_offset'])
            center_right_lane = int((min(arr) * w_min + max(arr) * w_max) / max(denom, 1e-6) + offset)
            error = int(image.shape[1] / 2) - center_right_lane
            return error * float(self.params['error_gain'])
        else:
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
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error

        P = error * p
        delta_t = time.time() - self.pre_t
        self.pre_t = time.time()
        delta_t = max(delta_t, float(self.params['pid_dt_epsilon']))
        D = (error - self.error_arr[1]) / delta_t * d
        I = np.sum(self.error_arr) * delta_t * i
        angle = P + I + D

        max_abs = float(self.params['pid_max_abs_angle'])
        if abs(angle) > max_abs:
            angle = np.sign(angle) * max_abs

        return int(angle)

    def calc_speed(self, angle):
        """
        Calculates the speed of the car based on the steering angle.

        Args:
        angle: The steering angle.

        Returns:
        The speed of the car.
        """
        a = abs(angle)
        t1 = float(self.params['speed_angle_low'])
        t2 = float(self.params['speed_angle_mid'])
        if a < t1:
            speed = int(self.params['speed_when_low'])
        elif t1 <= a <= t2:
            speed = int(self.params['speed_when_mid'])
        else:
            speed = int(self.params['speed_when_high'])
        return speed