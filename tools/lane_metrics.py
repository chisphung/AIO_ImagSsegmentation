import numpy as np
import math
from typing import Tuple


def compute_lane_metrics(segmented_image: np.ndarray,
                         sample_row: int = 62,
                         upper_row: int = 48,
                         min_width_px: int = 50,
                         center_bias_px: int = -10,
                         lane_value: int = 255) -> Tuple[float, float, float]:
    """
    Extract basic lane tracking metrics from a segmentation mask.
    Returns (e_y, e_psi, kappa_ref) approximations:
      - e_y: lateral error (pixels) from image center to right lane mid-point
      - e_psi: heading error (radians) as slope between two rows
      - kappa_ref: approximate path curvature (1/pixels) using three points
    """
    H, W = segmented_image.shape[:2]
    r0 = max(0, min(H-1, int(sample_row)))
    r1 = max(0, min(H-1, int(upper_row)))

    def lane_span(row_idx: int):
        xs = [x for x, y in enumerate(segmented_image[row_idx, :]) if y[0] == lane_value]
        if not xs:
            return None
        return min(xs), max(xs)

    span0 = lane_span(r0)
    span1 = lane_span(r1)

    # e_y: centerline to right-lane center at r0
    e_y = 0.0
    if span0:
        x_left, x_right = span0
        xc = int((x_left + 2.5 * x_right) / 3.5) + int(center_bias_px)
        e_y = (W/2.0) - xc

    # e_psi: approximate heading as angle of lane center between two rows
    e_psi = 0.0
    if span0 and span1 and r0 != r1:
        x_left0, x_right0 = span0
        xc0 = (x_left0 + 2.5 * x_right0) / 3.5 + center_bias_px
        x_left1, x_right1 = span1
        xc1 = (x_left1 + 2.5 * x_right1) / 3.5 + center_bias_px
        dy = float(r0 - r1)
        dx = float(xc0 - xc1)
        e_psi = math.atan2(dx, dy)  # note: image coords (y down)

    # kappa_ref: crude curvature using three points along the lane right-edge
    kappa_ref = 0.0
    if span0 and span1:
        x_left0, x_right0 = span0
        x_left1, x_right1 = span1
        # Three points: (x_right1, r1), (x_right0, r0), and middle row
        rm = max(0, min(H-1, (r0 + r1)//2))
        xs_m = [x for x, y in enumerate(segmented_image[rm, :]) if y[0] == lane_value]
        if xs_m:
            x_rightm = max(xs_m)
            p1 = np.array([x_right1, r1], dtype=float)
            p2 = np.array([x_right0, r0], dtype=float)
            p3 = np.array([x_rightm, rm], dtype=float)
            # Circle curvature approximation
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            s = 0.5 * (a + b + c)
            area = max(1e-6, np.sqrt(max(0.0, s * (s-a) * (s-b) * (s-c))))
            kappa_ref = 4 * area / max(1e-6, a*b*c)

    return float(e_y), float(e_psi), float(kappa_ref)
