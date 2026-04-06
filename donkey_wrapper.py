import gym
import gym_donkeycar
import numpy as np
from config.config import Config

class DonkeyEnvWrapper:
    def __init__(self):
        self.env = gym.make(
            Config.ENV_NAME,
            conf=Config.DONKEY_CONF
        )
        # minimum throttle to allow slowing down more in sharp turns
        self.min_throttle = 0.12  
        self.proxy_progress = 0.0
        
    def reset(self):
        reset_out = self.env.reset()
        # Support gymnasium returning (obs, info)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        self.proxy_progress = 0.0
        return self._process_observation(obs, info)
    
    def step(self, action):
        # Clip and scale actions to safe ranges
        steering = np.clip(action[0], -0.9, 0.9)    
        throttle = np.clip(action[1], 0.0, 1.0)   
        
        # Ensure minimum throttle to prevent getting stuck
        if throttle < self.min_throttle:
            throttle = self.min_throttle
        
        clipped_action = [float(steering), float(throttle)]
        
        # Execute action and support gym/gymnasium APIs
        step_out = self.env.step(clipped_action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            # gymnasium: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            # gym: (obs, reward, done, info)
            obs, reward, done, info = step_out
        
        # Process observation
        processed_obs = self._process_observation(obs, info)
        
        # Force termination on any collision with obstacles (cones/barriers)
        try:
            if info.get('hit', 'none') != 'none':
                done = True
        except Exception:
            pass
        
        return processed_obs, reward, done, info
    
    def _process_observation(self, obs, info):
        # Extract image
        if isinstance(obs, dict):
            image = obs.get('cam', obs.get('image', np.zeros((120, 160, 3))))
        else:
            image = obs
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Extract sensor data from obs first, then fall back to info
        if isinstance(obs, dict):
            speed = obs.get('speed', info.get('speed', 0.0))
            cte = obs.get('cte', info.get('cte', 0.0))
            steering_angle = obs.get('angle', info.get('angle', 0.0))
            progress_val = obs.get('progress', info.get('progress', None))
        else:
            speed = info.get('speed', 0.0)
            cte = info.get('cte', 0.0)
            steering_angle = info.get('angle', 0.0)
            progress_val = info.get('progress', None)
        
        # Get progress if available; otherwise approximate via integrated speed
        progress = progress_val
        if progress is None:
            progress = 0.0
        try:
            if (progress is None) or (float(progress) <= 0.0):
                dt = 0.05
                # Make proxy progress conservative and alignment-aware so straight-line off-track motion isn't over-rewarded
                track_len_est = 55.0
                # Alignment/center gating factor
                abs_angle = abs(float(steering_angle)) if steering_angle is not None else 0.0
                abs_cte = abs(float(cte)) if cte is not None else 0.0
                if abs_angle < 0.20 and abs_cte < 1.2:
                    gate = 1.0
                elif abs_angle < 0.35 and abs_cte < 2.0:
                    gate = 0.35
                else:
                    gate = 0.08
                inc = max(0.0, float(speed)) * dt / track_len_est * gate
                self.proxy_progress = min(1.0, self.proxy_progress + inc)
                progress = float(self.proxy_progress)
            else:
                self.proxy_progress = float(progress)
        except Exception:
            progress = float(self.proxy_progress)
        
        return {
            'image': image,
            'speed': speed,
            'cte': cte,
            'angle': steering_angle,
            'progress': progress
        }
    
    def set_min_throttle(self, value: float):
        self.min_throttle = float(np.clip(value, 0.0, 0.8))
    
    def close(self):
        self.env.close()