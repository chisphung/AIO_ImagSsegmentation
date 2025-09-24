from CEEC_Library import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time
import os
import math
import torch

from ultralytics import YOLO

from tools.custom import LandDetect
from tools.controllerV1 import Controller
from utils.config import ModelConfig, ControlConfig
from tools.segmentation import filter_masks_by_confidence, myGetSegment
from utils.socket import create_socket
from tools.lane_metrics import compute_lane_metrics
from tools.control_core import (
    PIDController,
    PIController,
    VehicleParams,
    LQRBicycleController,
    RateLimiter,
)

if __name__ == "__main__":
    # Create socket
    # Config model
    config_model = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Config control
    config_control = ControlConfig()

    # Controller
    controller = Controller()

    # Control mode: LEGACY (default), PID_ADV, or LQR
    CONTROL_MODE = os.environ.get('CONTROL_MODE', 'LEGACY').upper()

    # Vehicle/control params for advanced modes
    veh = VehicleParams()
    steer_rate_limiter_deg = RateLimiter(
        max_rate_up=veh.steer_rate_limit_dps,
        max_rate_down=veh.steer_rate_limit_dps,
    )

    # Advanced PID controller for steering (outputs degrees)
    # Uses normalized lateral error and preview heading term as input.
    pid_dt_default = 1.0 / 30.0
    pid_adv = PIDController(
        kp=4.0, ki=0.2, kd=3.0,
        dt=pid_dt_default,
        out_limits=(-veh.steer_limit_deg, veh.steer_limit_deg),
        i_limits=(-10.0, 10.0),
        derivative_filter_hz=5.0,
        anti_windup='backcalc', backcalc_beta=0.5,
    )

    # LQR controller (works in SI units internally)
    lqr_dt_default = 1.0 / 30.0
    lqr_ctrl = LQRBicycleController(veh=veh, dt=lqr_dt_default)

    # Load YOLOv8
    yolo = YOLO(config_model.weights_yolo, device).cuda()
    land_detector = YOLO(config_model.weights_lane).cuda()
    try:
        cnt_fps = 0
        t_pre = 0
        prev_loop_time = time.time()
        prev_angle_deg_cmd = 0.0
        prev_err_adv = 0.0

        # Advanced control tuning via env
        LM_SIGN = int(os.environ.get('LM_SIGN', '1'))  # flip lateral error sign if needed
        HM_SIGN = int(os.environ.get('HM_SIGN', '-1'))   # flip heading error sign (image coords)
        ADV_BLEND = float(os.environ.get('ADV_BLEND', '0.1'))  # blend with legacy [0..1]
        ADV_BLEND = max(0.0, min(1.0, ADV_BLEND))
        SPEED_TO_MPS = float(os.environ.get('SPEED_TO_MPS', '0.10'))  # scale current_speed -> m/s for LQR
        LQR_FF_GAIN = float(os.environ.get('LQR_FF_GAIN', '0.0'))  # 0.0 disables feedforward by default

        # Auto-calibration for sign selection
        CALIB_FRAMES = int(os.environ.get('ADV_CALIB_FRAMES', '40'))
        lm_sign_fix = None
        hm_sign_fix = None
        sign_counts = {(+1, +1): 0, (+1, -1): 0, (-1, +1): 0, (-1, -1): 0}
        adv_blend_runtime = ADV_BLEND
        sat_counter = 0
        # Track LQR behavior
        lqr_same_sign_count = 0
        lqr_last_sign = 0

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
            #     # Send signal to the server.
            #     message_getState = bytes("0", "utf-8")
            #     s.sendall(message_getState)

            #     # Set a timeout of 0.1 seconds.
            #     s.settimeout(0.1)

            #     # Receive 100 bytes of data from the server.
            #     state_date = s.recv(100)

            #     # Decode the received data from UTF-8.
            #     config_control.current_speed, config_control.current_angle = state_date.decode("utf-8").split(" ")

            # except Exception as er:
            #     pass

            # # Send sendBack_angle and sendBack_Speed to the server.
            # message = bytes(f"1 {config_control.sendBack_angle} {config_control.sendBack_Speed}", "utf-8")
            # s.sendall(message)

            # # Set a timeout of 0.5 seconds.
            # s.settimeout(0.5)

            # # Receive 100000 bytes of data from the server.
            # data = s.recv(100000)

            # try:
            #     config_control.current_speed = speed
            #     config_control.current_angle = angle
            # except:
            #     pass

            try:
                # Decode image in byte type recieved from server
                image = GetRaw()
                # ============================================================ PIDNet
                segmented_image = myGetSegment(image, land_detector)
                # segmented_image = cv2.resize(segmented_image, (320, 180))
                segmented_image = cv2.resize(segmented_image, (160, 80))
                # segmented_image = GetSeg()
                # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
                # ============================================================ YOLO
                # Resize the image to the desired dimensions
                # image = cv2.resize(image, (640, 384))

                with torch.no_grad():
                    yolo_output = yolo(image)[0]

                # # ============================================================ Controller
                angle, speed, next_step, mask_l, mask_r = controller.control(
                    segmented_image=segmented_image,
                    yolo_output=yolo_output,
                )

                # Control when turing
                if next_step:
                    # AVControl(speed = speed, angle = -angle)
                    print("Next step")
                    print("Angle:", angle)
                    print("Speed:", speed)
                    config_control.update(-angle, speed)

                    reset_counter = 1

                # Default control
                else:
                    # Measure loop dt
                    now = time.time()
                    dt = max(1e-3, now - prev_loop_time)
                    prev_loop_time = now

                    if CONTROL_MODE == 'LEGACY':
                        error = controller.calc_error(segmented_image)
                        angle = controller.PID(error, p=0.18, i=0.01, d=0.15)

                        # Speed scheduling similar to legacy behavior
                        if 1 <= reset_counter < 35:
                            speed = 25
                            reset_counter += 1
                        elif reset_counter == 35:
                            reset_counter = 0
                            speed = 25
                        else:
                            speed = controller.calc_speed(angle)
                            try:
                                if float(config_control.current_speed) > 44.5:
                                    speed = 15
                            except Exception:
                                pass

                        print("Error:", error)
                        print("Angle:", angle)
                        print("Speed:", speed)

                        config_control.update(-angle, speed)

                    elif CONTROL_MODE == 'PID_ADV':
                        # Compute lane metrics from segmentation
                        e_y_px, e_psi_rad, kappa_ref = compute_lane_metrics(
                            segmented_image,
                            sample_row=62,
                            upper_row=48,
                            center_bias_px=-10,
                            lane_value=255,
                        )

                        # Lane confidence gating at base row
                        H, W = segmented_image.shape[:2]
                        r0 = min(62, H-1)
                        xs = [x for x, y in enumerate(segmented_image[r0, :]) if y[0] == 255]
                        lane_width = (max(xs) - min(xs)) if len(xs) > 1 else 0
                        lane_conf_ok = (len(xs) >= 12 and lane_width >= 18)

                        # Fallback: blend with legacy angle if lane is weak
                        legacy_angle = controller.PID(controller.calc_error(segmented_image), p=0.18, i=0.01, d=0.15)

                        # Auto sign calibration during initial frames
                        lm_eff, hm_eff = LM_SIGN, HM_SIGN
                        if lm_sign_fix is None or hm_sign_fix is None:
                            # pick signs to align error direction with legacy steering sign
                            desired = 0.0
                            try:
                                desired = math.copysign(1.0, legacy_angle) if abs(legacy_angle) > 1e-3 else 0.0
                            except Exception:
                                desired = 0.0
                            best_pair = (LM_SIGN, HM_SIGN)
                            best_score = -1e9
                            for lm_cand in (+1, -1):
                                for hm_cand in (+1, -1):
                                    e_norm_c = lm_cand * (float(e_y_px) / max(1.0, (W / 2.0)))
                                    epsi_c = max(-0.5, min(0.5, hm_cand * float(e_psi_rad)))
                                    err_c = e_norm_c + 0.5 * epsi_c
                                    # score: larger when err sign matches desired and magnitude reasonable
                                    score = (err_c * desired) - 0.2 * abs(err_c)
                                    if score > best_score:
                                        best_score = score
                                        best_pair = (lm_cand, hm_cand)
                            lm_eff, hm_eff = best_pair
                            sign_counts[best_pair] += 1
                            CALIB_FRAMES -= 1
                            if CALIB_FRAMES <= 0:
                                # freeze to most frequent
                                lm_sign_fix, hm_sign_fix = max(sign_counts.items(), key=lambda kv: kv[1])[0]
                        else:
                            lm_eff, hm_eff = lm_sign_fix, hm_sign_fix

                        # Normalize lateral error by half image width with optional sign flip
                        e_norm = lm_eff * (float(e_y_px) / max(1.0, (W / 2.0)))

                        # Align heading term with lateral error to avoid wrong-direction bias
                        k_preview = 0.3
                        epsi_base = float(e_psi_rad)
                        # Choose heading sign to align with lateral error
                        hm_eff = 1 if e_norm == 0 else (1 if e_norm * epsi_base >= 0 else -1)
                        epsi_eff = hm_eff * epsi_base
                        epsi_eff = max(-0.5, min(0.5, epsi_eff))  # clamp to +/- ~29 deg
                        err = e_norm + k_preview * epsi_eff
                        if abs(err) < 0.02:
                            err = 0.0

                        # Smooth error (EMA)
                        alpha = 0.7
                        err_sm = alpha * err + (1.0 - alpha) * prev_err_adv
                        prev_err_adv = err_sm

                        # Update PID with tuned gains and current dt
                        pid_adv.kp = 6.0
                        pid_adv.ki = 0.05
                        pid_adv.kd = 1.0
                        pid_adv.dt = dt
                        angle_deg_cmd = pid_adv.update(err_sm)

                        # Enforce steering rate limits in deg/s
                        angle_deg_cmd = steer_rate_limiter_deg.limit(angle_deg_cmd, prev_angle_deg_cmd, dt)
                        prev_angle_deg_cmd = angle_deg_cmd

                        # Clamp to physical limits
                        angle_adv = max(-veh.steer_limit_deg, min(veh.steer_limit_deg, angle_deg_cmd))

                        # Blend with legacy if lane weak
                        # Saturation handling: if saturated for consecutive frames, reduce advanced weight
                        if abs(angle_adv) >= veh.steer_limit_deg - 0.5:
                            sat_counter += 1
                            if sat_counter >= 5:
                                adv_blend_runtime = max(0.3, adv_blend_runtime * 0.8)
                        else:
                            sat_counter = 0
                            adv_blend_runtime = min(ADV_BLEND, adv_blend_runtime + 0.02)

                        angle = (adv_blend_runtime * angle_adv + (1.0 - adv_blend_runtime) * legacy_angle) if lane_conf_ok else legacy_angle

                        # Final sanity: if angle fights lateral error strongly, damp it
                        if (angle * e_norm) < 0 and abs(e_norm) > 0.05:
                            angle *= 0.5

                        # Longitudinal: simple schedule based on steering demand
                        base_speed = 25
                        min_speed = 12
                        k_slow = 0.25  # speed reduction per deg
                        speed = max(min_speed, base_speed - k_slow * abs(angle))

                        if 1 <= reset_counter < 35:
                            speed = 25
                            reset_counter += 1
                        elif reset_counter == 35:
                            reset_counter = 0

                        print(f"PID_ADV e_y={e_y_px:.1f}px e_psi={math.degrees(e_psi_rad):.2f}deg -> angle={angle:.2f} deg speed={speed:.1f}")
                        config_control.update(-angle, speed)

                    elif CONTROL_MODE == 'LQR':
                        # Compute lane metrics
                        e_y_px, e_psi_rad, kappa_ref = compute_lane_metrics(
                            segmented_image,
                            sample_row=62,
                            upper_row=48,
                            center_bias_px=-10,
                            lane_value=255,
                        )

                        # Lane confidence gating at base row
                        H, W = segmented_image.shape[:2]
                        r0 = min(62, H-1)
                        xs = [x for x, y in enumerate(segmented_image[r0, :]) if y[0] == 255]
                        lane_width = (max(xs) - min(xs)) if len(xs) > 1 else 0
                        lane_conf_ok = (len(xs) >= 12 and lane_width >= 18)

                        # Legacy fallback for blending
                        legacy_angle = controller.PID(controller.calc_error(segmented_image), p=0.18, i=0.01, d=0.15)

                        # Auto sign calibration similar to PID branch (use proxy error)
                        lm_eff, hm_eff = LM_SIGN, HM_SIGN
                        if lm_sign_fix is None or hm_sign_fix is None:
                            desired = 0.0
                            try:
                                desired = math.copysign(1.0, legacy_angle) if abs(legacy_angle) > 1e-3 else 0.0
                            except Exception:
                                desired = 0.0
                            best_pair = (LM_SIGN, HM_SIGN)
                            best_score = -1e9
                            for lm_cand in (+1, -1):
                                for hm_cand in (+1, -1):
                                    e_norm_c = lm_cand * (float(e_y_px) / max(1.0, (W / 2.0)))
                                    epsi_c = max(-0.5, min(0.5, hm_cand * float(e_psi_rad)))
                                    err_c = e_norm_c + 0.5 * epsi_c
                                    score = (err_c * desired) - 0.2 * abs(err_c)
                                    if score > best_score:
                                        best_score = score
                                        best_pair = (lm_cand, hm_cand)
                            lm_eff, hm_eff = best_pair
                            sign_counts[best_pair] += 1
                            CALIB_FRAMES -= 1
                            if CALIB_FRAMES <= 0:
                                lm_sign_fix, hm_sign_fix = max(sign_counts.items(), key=lambda kv: kv[1])[0]
                        else:
                            lm_eff, hm_eff = lm_sign_fix, hm_sign_fix

                        # Pixel-to-meter scale assumption (approx.)
                        scale_px_to_m = 0.8 / 160.0  # 160px ~ 0.8m across
                        e_y_m = lm_eff * float(e_y_px) * scale_px_to_m
                        kappa_m = float(kappa_ref) / max(1e-6, scale_px_to_m)

                        # Vehicle speed (m/s): derive from current_speed via scaling, clamp range
                        try:
                            v_mps = float(config_control.current_speed) * SPEED_TO_MPS
                        except Exception:
                            v_mps = 3.0
                        v_mps = max(1.0, min(6.0, v_mps))

                        # Heavier penalty on lateral error
                        try:
                            import numpy as np
                            lqr_ctrl.Q = np.diag([4.0, 1.5])
                        except Exception:
                            pass

                        # Align heading term with lateral error
                        epsi_base = float(e_psi_rad)
                        hm_eff = 1 if e_y_m == 0 else (1 if (e_y_m * epsi_base) >= 0 else -1)
                        epsi_eff = hm_eff * epsi_base
                        epsi_eff = max(-0.5, min(0.5, epsi_eff))

                        # If lateral error is tiny, suppress heading to avoid bias drift
                        e_norm_tmp = lm_eff * (float(e_y_px) / max(1.0, (W / 2.0)))
                        if abs(e_norm_tmp) < 0.02:
                            epsi_eff = 0.0

                        # Feedforward curvature: align sign with heading and scale by gain
                        kappa_m_eff = 0.0 if LQR_FF_GAIN == 0.0 else (LQR_FF_GAIN * math.copysign(abs(kappa_m), epsi_eff))

                        # LQR update returns radians
                        delta_rad = lqr_ctrl.update(e_y=e_y_m, e_psi=epsi_eff, v=v_mps, kappa_ref=kappa_m_eff)
                        angle_deg_cmd = math.degrees(delta_rad)

                        # Rate limit in deg/s
                        angle_deg_cmd = steer_rate_limiter_deg.limit(angle_deg_cmd, prev_angle_deg_cmd, dt)
                        prev_angle_deg_cmd = angle_deg_cmd

                        angle_adv = max(-veh.steer_limit_deg, min(veh.steer_limit_deg, angle_deg_cmd))

                        # Saturation handling & adaptive blending
                        if abs(angle_adv) >= veh.steer_limit_deg - 0.5:
                            sat_counter += 1
                            if sat_counter >= 5:
                                adv_blend_runtime = max(0.3, adv_blend_runtime * 0.8)
                        else:
                            sat_counter = 0
                            adv_blend_runtime = min(ADV_BLEND, adv_blend_runtime + 0.02)

                        angle = (adv_blend_runtime * angle_adv + (1.0 - adv_blend_runtime) * legacy_angle) if lane_conf_ok else legacy_angle

                        # If advanced keeps same turn sign with near-zero lateral error, reduce advanced weight
                        sign_adv = 0
                        if angle_adv > 1.0:
                            sign_adv = 1
                        elif angle_adv < -1.0:
                            sign_adv = -1
                        if abs(e_norm_tmp) < 0.05 and sign_adv != 0:
                            if sign_adv == lqr_last_sign:
                                lqr_same_sign_count += 1
                                if lqr_same_sign_count >= 8:
                                    adv_blend_runtime = max(0.3, adv_blend_runtime * 0.8)
                            else:
                                lqr_same_sign_count = 0
                            lqr_last_sign = sign_adv

                        # Final sanity: damp if angle opposes lateral error (use effective sign)
                        e_norm = lm_eff * (float(e_y_px) / max(1.0, (W / 2.0)))
                        if (angle * e_norm) < 0 and abs(e_norm) > 0.05:
                            angle *= 0.5

                        # Longitudinal schedule similar to PID_ADV
                        base_speed = 25
                        min_speed = 12
                        k_slow = 0.25
                        speed = max(min_speed, base_speed - k_slow * abs(angle))

                        if 1 <= reset_counter < 35:
                            speed = 25
                            reset_counter += 1
                        elif reset_counter == 35:
                            reset_counter = 0

                        print(f"LQR e_y={e_y_m:.2f}m e_psi={math.degrees(e_psi_rad):.2f}deg -> angle={angle:.2f} deg speed={speed:.1f}")
                        config_control.update(-angle, speed)

                    else:
                        # Fallback to legacy if mode unrecognized
                        error = controller.calc_error(segmented_image)
                        angle = controller.PID(error, p=0.18, i=0.01, d=0.15)
                        speed = controller.calc_speed(angle)
                        config_control.update(-angle, speed)

                AVControl(speed=speed, angle=-angle)

                yolo_output = yolo_output.plot()

                if config_model.view_seg:
                    cv2.imshow("segmented image", segmented_image)

                if config_model.view_first_view:
                    cv2.imshow("first view image", yolo_output)

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
