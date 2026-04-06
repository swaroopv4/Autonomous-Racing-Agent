import numpy as np

class WarmStart:
    def __init__(self, warmup_steps=20):  
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def get_action(self, step, cte=0.0):
        if step < self.warmup_steps:
            # Smoother acceleration curve (start lower, ramp gentler)
            progress_ratio = step / self.warmup_steps
            throttle = 0.10 + (progress_ratio ** 1.5) * 0.35  

            # Slight center-seeking bias based on current CTE
            steer_bias = 0.0 if cte == 0 else -np.sign(cte) * min(0.18, abs(cte) * 0.12)
            
            return np.array([
                steer_bias,
                throttle
            ])
        
        return None  # Warmup done, use policy
    
    def reset(self):
        self.current_step = 0