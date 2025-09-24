import math
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class RateLimiter:
    def __init__(self, max_rate_up: float, max_rate_down: float):
        self.max_rate_up = float(max_rate_up)
        self.max_rate_down = float(max_rate_down)

    def limit(self, cmd: float, prev_cmd: float, dt: float) -> float:
        if dt <= 0:
            return cmd
        delta = cmd - prev_cmd
        max_up = self.max_rate_up * dt
        max_down = self.max_rate_down * dt
        if delta > max_up:
            return prev_cmd + max_up
        if delta < -max_down:
            return prev_cmd - max_down
        return cmd


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 dt: float,
                 out_limits: Tuple[float, float] = (-math.inf, math.inf),
                 i_limits: Tuple[float, float] = (-math.inf, math.inf),
                 derivative_filter_hz: float = 0.0,
                 anti_windup: str = 'clamp',  # 'clamp' or 'backcalc'
                 backcalc_beta: float = 0.5):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.out_min, self.out_max = out_limits
        self.i_min, self.i_max = i_limits
        self.alpha = 0.0
        if derivative_filter_hz > 0:
            # First-order low-pass for derivative: alpha = dt / (dt + 1/(2*pi*hz))
            tau = 1.0 / (2.0 * math.pi * float(derivative_filter_hz))
            self.alpha = self.dt / (self.dt + tau)
        self.anti_windup = anti_windup
        self.beta = float(backcalc_beta)

        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_derivative = 0.0

    def reset(self):
        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_derivative = 0.0

    def update(self, error: float, measurement: Optional[float] = None) -> float:
        # Proportional
        P = self.kp * error

        # Integral with clamping
        self.integrator += self.ki * error * self.dt
        self.integrator = max(self.i_min, min(self.i_max, self.integrator))
        I = self.integrator

        # Derivative on measurement if provided, else on error
        if measurement is not None and self.prev_measurement is not None:
            d_raw = -(measurement - self.prev_measurement) / max(self.dt, 1e-6)
        else:
            d_raw = (error - self.prev_error) / max(self.dt, 1e-6)
        D = self.kd * (self.alpha * d_raw + (1 - self.alpha) * self.prev_derivative)

        # Unsaturated output
        u = P + I + D
        u_sat = max(self.out_min, min(self.out_max, u))

        # Anti-windup back-calculation
        if self.anti_windup == 'backcalc':
            self.integrator += self.beta * (u_sat - u)
            self.integrator = max(self.i_min, min(self.i_max, self.integrator))

        # Save state
        self.prev_error = error
        if measurement is not None:
            self.prev_measurement = measurement
        self.prev_derivative = d_raw

        return u_sat


class PIController(PIDController):
    def __init__(self, kp: float, ki: float, dt: float,
                 out_limits: Tuple[float, float] = (-math.inf, math.inf),
                 i_limits: Tuple[float, float] = (-math.inf, math.inf),
                 anti_windup: str = 'clamp', backcalc_beta: float = 0.5):
        super().__init__(kp, ki, 0.0, dt, out_limits, i_limits, 0.0, anti_windup, backcalc_beta)


@dataclass
class VehicleParams:
    L: float = 0.25  # wheelbase in meters (example)
    steer_limit_deg: float = 30.0
    steer_rate_limit_dps: float = 300.0


class LQRBicycleController:
    """
    Discrete-time LQR on a simple lateral error model:
        e_y[k+1] = e_y[k] + v*dt * e_psi[k]
        e_psi[k+1] = e_psi[k] + (v*dt/L) * (delta[k] - delta_ff)
    Feed-forward: delta_ff = atan(L * kappa_ref)
    Control: delta = delta_ff - K @ [e_y, e_psi]^T
    """
    def __init__(self, veh: VehicleParams, dt: float, Q: Optional[np.ndarray] = None, R: float = 1.0):
        self.veh = veh
        self.dt = float(dt)
        self.Q = Q if Q is not None else np.diag([2.0, 1.0])
        self.R = float(R)
        self.K = np.zeros((1, 2))
        self.prev_delta = 0.0
        self.rate_limiter = RateLimiter(
            max_rate_up=self.veh.steer_rate_limit_dps * math.pi/180.0,
            max_rate_down=self.veh.steer_rate_limit_dps * math.pi/180.0,
        )

    def _dlqr(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: float) -> np.ndarray:
        # Solve discrete LQR via Riccati equation
        X = np.copy(Q)
        for _ in range(100):
            Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
            if np.allclose(X, Xn, atol=1e-6, rtol=1e-6):
                break
            X = Xn
        K = np.linalg.inv(R + B.T @ X @ B) @ (B.T @ X @ A)
        return K

    def update(self, e_y: float, e_psi: float, v: float, kappa_ref: float) -> float:
        v = float(max(0.01, v))
        L = float(self.veh.L)
        dt = self.dt

        # Linearized discrete model
        A = np.array([[1.0, v*dt],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [v*dt/L]])
        # Compute gain for current speed
        self.K = self._dlqr(A, B, self.Q, self.R)

        # Feed-forward steering for curvature (radians)
        delta_ff = math.atan(L * float(kappa_ref))

        x = np.array([[e_y], [e_psi]])
        delta_cmd = float(delta_ff - (self.K @ x).item())

        # Saturate steering angle
        steer_lim = math.radians(self.veh.steer_limit_deg)
        delta_cmd = max(-steer_lim, min(steer_lim, delta_cmd))

        # Rate limit
        delta_cmd = self.rate_limiter.limit(delta_cmd, self.prev_delta, dt)
        self.prev_delta = delta_cmd

        return delta_cmd  # radians
