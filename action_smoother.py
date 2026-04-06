import numpy as np
from collections import deque

class ActionSmoother:
    def __init__(self, window_size=3, alpha=0.6, max_steer_rate=0.15, max_throttle_rate=0.10):
        self.window_size = window_size
        self.alpha = alpha
        self.action_history = deque(maxlen=window_size)
        self.prev_action = None
        self.max_steer_rate = max_steer_rate
        self.max_throttle_rate = max_throttle_rate
        
    def smooth(self, action):
        # Add to history
        self.action_history.append(action.copy())
        
        if self.prev_action is None:
            self.prev_action = action
            return action
        
        # Exponential moving average
        smoothed = self.alpha * self.prev_action + (1 - self.alpha) * action
        
        # Additional smoothing using window average
        if len(self.action_history) >= 2:
            window_avg = np.mean(list(self.action_history), axis=0)
            smoothed = 0.7 * smoothed + 0.3 * window_avg
        
        # Apply per-dimension rate limiting relative to previous action
        delta = smoothed - self.prev_action
        # Steering index 0, throttle index 1
        steer_change = np.clip(delta[0], -self.max_steer_rate, self.max_steer_rate)
        throttle_change = np.clip(delta[1], -self.max_throttle_rate, self.max_throttle_rate)
        limited = self.prev_action + np.array([steer_change, throttle_change], dtype=np.float32)
        
        self.prev_action = limited
        return limited
    
    def reset(self):
        self.action_history.clear()
        self.prev_action = None

    def set_rate_limits(self, max_steer_rate=None, max_throttle_rate=None):
        if max_steer_rate is not None:
            self.max_steer_rate = float(max(0.0, max_steer_rate))
        if max_throttle_rate is not None:
            self.max_throttle_rate = float(max(0.0, max_throttle_rate))