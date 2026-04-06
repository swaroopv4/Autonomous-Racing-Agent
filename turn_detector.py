import numpy as np
from collections import deque

class TurnDetector:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.cte_history = deque(maxlen=history_size)
        self.angle_history = deque(maxlen=history_size)
        self.progress_history = deque(maxlen=history_size)
        self.post_turn_cooldown = 0
        self.cooldown_max = 14
        self.last_turn_direction = 'none'
    
    def reset(self):
        self.cte_history.clear()
        self.angle_history.clear()
        self.progress_history.clear()
        self.post_turn_cooldown = 0
        self.last_turn_direction = 'none'
    
    def update(self, cte, angle, progress):
        self.cte_history.append(cte)
        self.angle_history.append(angle)
        self.progress_history.append(progress)
    
    def detect_turn_type(self):
        if len(self.angle_history) < 5 or len(self.cte_history) < 5:
            return 'straight', 'none', 0.0
        # Analyze recent angles and CTE dynamics
        recent_angles = list(self.angle_history)[-5:]
        recent_cte = list(self.cte_history)[-5:]
        avg_abs_angle = np.mean(np.abs(recent_angles))
        mean_angle = np.mean(recent_angles)
        cte_slope = recent_cte[-1] - recent_cte[0]
        cte_var = np.var(recent_cte)
        cte_range = np.max(recent_cte) - np.min(recent_cte)

        # Determine turn direction: prefer angle sign, fallback to CTE slope
        if mean_angle > 0.01:
            direction = 'right'
        elif mean_angle < -0.01:
            direction = 'left'
        else:
            if cte_slope > 0.02:
                direction = 'right'
            elif cte_slope < -0.02:
                direction = 'left'
            else:
                direction = 'none'

        if direction != 'none':
            self.last_turn_direction = direction

        # Composite curvature metric from angle magnitude and CTE dynamics
        # Angle often near-zero in some wrappers; use CTE variance/range as proxy
        curvature = max(
            avg_abs_angle * 2.0,
            np.sqrt(max(0.0, cte_var)) * 0.8,
            abs(cte_slope) * 0.6,
            cte_range * 0.30
        )

        # Confidence scaled to [0,1]
        confidence = float(max(0.0, min(1.0, curvature * 6.0)))

        # Classify sharpness based on composite curvature
        if curvature < 0.01:
            return 'straight', direction, confidence
        elif curvature < 0.03:
            return 'gentle', direction, confidence
        elif curvature < 0.07:
            return 'medium', direction, confidence
        elif curvature < 0.12:
            return 'sharp', direction, confidence
        else:
            return 'hairpin', direction, confidence
    
    def get_steering_multiplier(self):
        turn_type, _, confidence = self.detect_turn_type()
        multipliers = {
            'straight': 1.0,
            'gentle': 1.02,
            'medium': 1.08,
            'sharp': 1.15,
            'hairpin': 1.20
        }
        base_multiplier = multipliers.get(turn_type, 1.0)
        # Blend with confidence
        return 1.0 + (base_multiplier - 1.0) * confidence
    
    def should_slow_down(self):
        turn_type, _, confidence = self.detect_turn_type()
        immediate_slow = False
        target_speed = 6.0
        if turn_type == 'hairpin':
            cap = 0.35  
        elif turn_type == 'sharp':
            cap = 0.45 
        elif turn_type == 'medium':
            cap = 0.60  
        else:
            cap = 0.95
        if turn_type == 'hairpin' and confidence > 0.7:
            immediate_slow = True
            target_speed = 3.0 
        elif turn_type == 'sharp' and confidence > 0.6:
            immediate_slow = True
            target_speed = 4.0  
        elif turn_type == 'medium' and confidence > 0.5:
            immediate_slow = True
            target_speed = 4.5

        if immediate_slow:
            self.post_turn_cooldown = self.cooldown_max
            return True, target_speed

        if self.post_turn_cooldown > 0:
            frac = 1.0 - (self.post_turn_cooldown / self.cooldown_max)
            ramp_speed = 4.0 + (6.0 - 4.0) * frac
            self.post_turn_cooldown -= 1
            return True, ramp_speed

        return False, 6.0

    def is_in_cooldown(self):
        return self.post_turn_cooldown > 0

    def get_last_turn_direction(self):
        return self.last_turn_direction

    def throttle_cap(self, turn_type: str, should_slow: bool, target_speed: float) -> float:
        if turn_type == 'hairpin':
            cap = 0.35
        elif turn_type == 'sharp':
            cap = 0.45
        elif turn_type == 'medium':
            cap = 0.60
        else:
            cap = 0.95
        if should_slow:
            if target_speed <= 2.5:
                cap = min(cap, 0.35) 
            elif target_speed <= 3.5:
                cap = min(cap, 0.45) 
            elif target_speed <= 4.8:
                cap = min(cap, 0.55) 
            else:
                cap = min(cap, 0.70) 
            cap = max(0.0, cap - 0.05)
        return float(cap)

    def steering_post_turn_bias(self) -> float:
        last_dir = self.get_last_turn_direction()
        if last_dir == 'right':
            return 0.09
        if last_dir == 'left':
            return -0.09
        return 0.0

    def recommend_rate_limits(self, turn_type: str, in_cooldown: bool, cte_abs: float, speed_mode_steps: int):        # Cooldown: quicker steering, slower throttle changes
        if in_cooldown:
            max_steer_rate, max_throttle_rate = 0.38, 0.08
        else:
            if turn_type == 'hairpin':
                max_steer_rate, max_throttle_rate = 0.45, 0.12
            elif turn_type == 'sharp':
                max_steer_rate, max_throttle_rate = 0.40, 0.12
            elif turn_type == 'medium':
                max_steer_rate, max_throttle_rate = 0.30, 0.10
            else:
                max_steer_rate, max_throttle_rate = 0.30, 0.08
        # In speed mode on easier segments, allow faster throttle ramp
        if speed_mode_steps > 0 and turn_type in ['straight', 'gentle', 'medium']:
            max_steer_rate, max_throttle_rate = 0.30, 0.15
        # If far off center, relax steering to recover faster
        if cte_abs > 1.6:
            max_steer_rate, max_throttle_rate = 0.75, 0.10
        elif cte_abs > 1.2:
            max_steer_rate, max_throttle_rate = 0.60, 0.10
        return float(max_steer_rate), float(max_throttle_rate)
