from config.config import Config
import numpy as np

class RewardShaper:
    def __init__(self):
        self.prev_progress = 0
        self.prev_speed = 0
        self.step_count = 0
        self.max_progress = 0
        self.stuck_counter = 0
        self.prev_cte = 0
        self.off_track_counter = 0
        self.on_track_counter = 0
        self.milestone_10 = False
        self.milestone_25 = False
        self.milestone_50 = False
        self.milestone_75 = False
        self.prev_steering = 0
        self.steering_oscillation = 0
        self.no_progress_counter = 0
        self.prev_angle = 0
        self.is_offtrack = False
        self.offtrack_steps = 0
        self.offtrack_max_cte = 0.0
        
    def reset(self):
        self.prev_progress = 0
        self.prev_speed = 0
        self.step_count = 0
        self.max_progress = 0
        self.stuck_counter = 0
        self.prev_cte = 0
        self.off_track_counter = 0
        self.on_track_counter = 0
        self.milestone_10 = False
        self.milestone_25 = False
        self.milestone_50 = False
        self.milestone_75 = False
        self.prev_steering = 0
        self.steering_oscillation = 0
        self.no_progress_counter = 0
        self.prev_angle = 0
        self.is_offtrack = False
        self.offtrack_steps = 0
        self.offtrack_max_cte = 0.0
        
    def reset_lap_milestones(self):
        self.milestone_10 = False
        self.milestone_25 = False
        self.milestone_50 = False
        self.milestone_75 = False
        self.prev_progress = 0
        
    def calculate_reward(self, obs, done, info, turn_type='straight', turn_direction='none'):
        reward = 0.0
        self.step_count += 1
        
        speed = obs.get('speed', info.get('speed', 0))
        cte = obs.get('cte', info.get('cte', 0))
        progress = obs.get('progress', info.get('progress', 0))
        hit = info.get('hit', 'none')
        angle = obs.get('angle', info.get('angle', 0))
        
        # Track maximum progress achieved
        self.max_progress = max(self.max_progress, progress)
        start_grace = (self.step_count < 250 and self.max_progress < 0.08)
        
        # PROGRESS REWARD 
        progress_delta = progress - self.prev_progress
        
        if progress_delta > 0.0001:  
            # MASSIVE exponential reward for progress
            base_progress_reward = progress_delta * 2000.0  
            
            # Bonus multiplier based on how far you've gotten
            progress_multiplier = 1.0 + (progress * 3.0)  
            reward += base_progress_reward * progress_multiplier
            
            # Reset counters
            self.stuck_counter = 0
            self.no_progress_counter = 0
            
            # Extra bonus for big jumps in progress
            if progress_delta > 0.02:
                reward += 200.0  
            
            if progress_delta > 0.01:  
                reward += 100.0  
                
        elif progress_delta < -0.001:  
            reward -= 20.0 
            
        else:  
            self.no_progress_counter += 1
            self.stuck_counter += 1
            reward -= 2.0  
        
        self.prev_progress = progress

        # CENTERING NUDGES 
        cte_abs = abs(cte)
        prev_cte_abs = abs(self.prev_cte)
        if cte_abs > 0.8:
            wide_penalty = min(60.0, (cte_abs - 0.8) ** 2 * 40.0)
            reward -= wide_penalty
        if prev_cte_abs - cte_abs > 0.05:
            reward += 8.0
        
        # MILESTONE BONUSES (INCREASED) 
        if progress > 0.10 and not self.milestone_10:
            reward += 500.0  
            self.milestone_10 = True
            if Config.VERBOSE:
                print(f"10% Checkpoint! (+500)")
            
        if progress > 0.25 and not self.milestone_25:
            reward += 800.0  
            self.milestone_25 = True
            if Config.VERBOSE:
                print(f"25% Checkpoint! (+800)")
            
        if progress > 0.50 and not self.milestone_50:
            reward += 1500.0  
            self.milestone_50 = True
            if Config.VERBOSE:
                print(f"50% Checkpoint! (+1500)")
            
        if progress > 0.75 and not self.milestone_75:
            reward += 2500.0  
            self.milestone_75 = True
            if Config.VERBOSE:
                print(f"75% Checkpoint! (+2500)")
        
        # CTE HANDLING 
        abs_cte = abs(cte)
        
        # CRITICAL CHANGE: Only penalize CTE if NOT making progress
        if progress_delta <= 0.0001:  
            if abs_cte < 1.0:  
                self.on_track_counter += 1
                self.off_track_counter = 0
                reward += 5.0  
                
            elif abs_cte < 2.0:  
                self.off_track_counter += 1
                reward -= 3.0  
                
            elif abs_cte < 3.0:  
                self.off_track_counter += 1
                reward -= 8.0  
                
            else:  
                self.off_track_counter += 1
                reward -= 15.0 
        else:
            self.on_track_counter += 1
            self.off_track_counter = 0
            
            if abs_cte < 1.0:
                reward += 10.0
        
        if cte_abs > 1.0:
            if not self.is_offtrack:
                self.is_offtrack = True
                self.offtrack_steps = 0
                self.offtrack_max_cte = cte_abs
            else:
                self.offtrack_max_cte = max(self.offtrack_max_cte, cte_abs)
            self.offtrack_steps += 1
            if cte_abs > 2.5:
                reward -= 12.0
            elif cte_abs > 1.8:
                reward -= 8.0
            else:
                reward -= 5.0
        else:
            if self.is_offtrack and cte_abs < 0.8:
                one_time = 20.0 + 2.0 * float(self.offtrack_steps) + 10.0 * max(0.0, self.offtrack_max_cte - 1.0)
                reward -= one_time
                self.is_offtrack = False
                self.offtrack_steps = 0
                self.offtrack_max_cte = 0.0

        # SPEED REWARD 
        if progress_delta > 0.0001:
            reward += speed * 5.0
            
            if speed > 4.0:
                reward += 20.0
            
        else:
            if speed > 2.0:
                reward -= 3.0
        
        # ALIGNMENT REWARD 
        abs_angle = abs(angle)
        
        if progress_delta > 0.0001:  
            if abs_angle < 0.1:  
                reward += 8.0
            elif abs_angle < 0.3:  
                reward += 4.0
        
        # TURNING ENCOURAGEMENT 
        if progress_delta > 0.005 and abs_cte > 1.5:
            reward += 50.0
            if Config.VERBOSE and self.step_count % 10 == 0:
                print(f"Taking turn! (CTE: {abs_cte:.2f}, Progress: +{progress_delta:.4f})")
        
        # COLLISION PENALTY 
        if hit != 'none':
            # Scale penalty based on progress achieved
            if progress < 0.05:
                penalty = 50.0  # Very light penalty for early collisions
            elif progress < 0.15:
                penalty = 100.0  
            elif progress < 0.30:
                penalty = 200.0  # Medium penalty
            else:
                penalty = 400.0  # Full penalty for late collisions
            reward -= penalty
            if Config.VERBOSE:
                print(f"Collision at {progress:.3f} (-{penalty})")
        
        # STUCK DETECTION 
        if self.no_progress_counter > 40 and not start_grace:  # Stuck for 40 steps
            reward -= 100.0
            if Config.VERBOSE and self.step_count % 50 == 0:
                print(f"Stuck at {self.max_progress:.3f}")
        
        # OFF-TRACK TOO LONG 
        if self.off_track_counter > 50 and progress_delta <= 0.0001 and not start_grace:
            reward -= 80.0
            if Config.VERBOSE and self.step_count % 50 == 0:
                print(f"Off track too long without progress")
        
        # EARLY FAILURE DETECTION 
        if self.step_count > 100 and self.max_progress < 0.03:
            reward -= 150.0
            if Config.VERBOSE and self.step_count % 50 == 0:
                print(f"Failed to make early progress")
        
        # LAP COMPLETION 
        if done and hit == 'none':
            if progress > 0.95:
                reward += 10000.0  # MASSIVE bonus
                if Config.VERBOSE:
                    print(f"LAP COMPLETE! (+10000)")
            elif progress > 0.75:
                reward += 2000.0
            elif progress > 0.50:
                reward += 1000.0
            elif progress > 0.25:
                reward += 400.0
            elif progress > 0.10:
                reward += 150.0
        
        # MINIMAL TIME PENALTY 
        reward -= 0.02  
        
        # Store previous values
        self.prev_angle = angle
        
        return reward
    
    def get_info(self):
        return {
            'step_count': self.step_count,
            'max_progress': self.max_progress,
            'stuck_counter': self.stuck_counter,
            'no_progress_counter': self.no_progress_counter,
            'off_track_counter': self.off_track_counter,
            'on_track_counter': self.on_track_counter
        }