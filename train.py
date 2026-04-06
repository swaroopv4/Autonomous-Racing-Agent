import torch
import numpy as np
import os
import json
import sys
import imageio
import cv2
import threading
import time

try:
    import mss
    MSS_AVAILABLE = True
except Exception:
    mss = None
    MSS_AVAILABLE = False

from config.config import Config
from models.network import CNNActorCritic
from models.ppo import PPOTrainer
from environments.donkey_wrapper import DonkeyEnvWrapper
from utils.reward_shaper import RewardShaper
from utils.turn_detector import TurnDetector
from utils.warm_start import WarmStart
from utils.action_smoother import ActionSmoother
from utils.curriculum import Curriculum

import matplotlib.pyplot as plt


class ImprovedPPOTrainer(PPOTrainer):
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)
        self.base_entropy_coef = Config.ENTROPY_COEF
        
    def set_entropy_coef(self, entropy_coef):
        self.entropy_coef = entropy_coef


class ScreenRecorder:
    def __init__(self, monitor_index=1, fps=40, output_path=None):
        self.monitor_index = int(monitor_index)
        self.fps = max(1, int(fps))
        self.output_path = output_path or os.path.join(Config.RESULTS_DIR, "unity_screen.mp4")
        self._stop = threading.Event()
        self._thread = None
        self._writer = None
        self._sct = None
        self._ready = False

    def start(self):
        if not MSS_AVAILABLE:
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ScreenRecorder", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        try:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._writer:
                self._writer.release()
        except Exception:
            pass
        try:
            if self._sct:
                self._sct.close()
        except Exception:
            pass

    def _run(self):
        try:
            self._sct = mss.mss()
            monitors = self._sct.monitors
            mon_idx = self.monitor_index if 0 <= self.monitor_index < len(monitors) else 1
            mon = monitors[mon_idx]
            bbox = {
                "top": mon["top"],
                "left": mon["left"],
                "width": mon["width"],
                "height": mon["height"],
            }
            size = (bbox["width"], bbox["height"])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, float(self.fps), size)
            self._ready = True
            frame_time = 1.0 / float(self.fps)
            while not self._stop.is_set():
                t0 = time.time()
                img = self._sct.grab(bbox)
                # mss returns BGRA; convert to BGR
                frame = np.array(img)[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self._writer.write(frame)
                dt = time.time() - t0
                if dt < frame_time:
                    time.sleep(frame_time - dt)
        except Exception:
            pass

def train(resume_from=None):
    
    print("=" * 70)
    print("PPO Racing Agent Training")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Optional: start full-screen recorder
    screen_recorder = None
    if getattr(Config, 'RECORD_SCREEN', False):
        if MSS_AVAILABLE:
            try:
                screen_recorder = ScreenRecorder(
                    monitor_index=getattr(Config, 'SCREEN_MONITOR', 1),
                    fps=getattr(Config, 'SCREEN_FPS', 40),
                    output_path=getattr(Config, 'SCREEN_OUTPUT', None)
                )
                if screen_recorder.start():
                    print(f"Screen recording started -> {screen_recorder.output_path} @ {screen_recorder.fps} FPS (monitor {screen_recorder.monitor_index})")
                else:
                    print("Screen recorder did not start (unknown reason)")
            except Exception as _e:
                print("Could not start screen recorder (permissions or setup). Continuing without.")
        else:
            print("Screen recording disabled: 'mss' not available. Run 'pip install mss' to enable.")

    # Create environment
    print("Initializing environment...")
    print("Make sure the Donkey Simulator is running!")
    
    try:
        env = DonkeyEnvWrapper()
        print(" Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        sys.exit(1)
    
    # Create model
    print("Creating neural network...")
    model = CNNActorCritic(action_dim=Config.ACTION_DIM)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("Initializing PPO trainer...")
    trainer = ImprovedPPOTrainer(model, device=device)
    
    # Create all helper modules
    print("Initializing helper modules...")
    curriculum = Curriculum()
    reward_shaper = RewardShaper()
    turn_detector = TurnDetector(history_size=10)
    warm_start = WarmStart(warmup_steps=20)
    action_smoother = ActionSmoother(window_size=3, alpha=0.45)
    print("Curriculum learning")
    print("Turn detector")
    print("Warm start")
    print("Action smoother")
    
    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nLoading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            import re
            match = re.search(r'episode_(\d+)', resume_from)
            if match:
                start_episode = int(match.group(1))
                print(f"Resuming from episode {start_episode}")
        except:
            print("Checkpoint loaded")
    
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    lap_times = []
    success_count = 0
    stuck_count = 0
    collision_count = 0
    turn_type_stats = {'straight': 0, 'gentle': 0, 'medium': 0, 'sharp': 0, 'hairpin': 0}
    best_lap_time = float('inf')
    last_lap_count = 0
    episode_successes = []
    total_laps_completed = 0
    # Learned per-segment throttle profile (progress-binned)
    PROFILE_BINS = 50
    speed_profile = np.zeros(PROFILE_BINS, dtype=np.float32)
    profile_counts = np.zeros(PROFILE_BINS, dtype=np.int32)
    
    # Best model tracking
    best_progress = 0.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pt")
    best_progress_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_progress.pt")
    
    # Training loop
    print(f"\nStarting training for {Config.NUM_EPISODES} episodes...")
    print("=" * 70)
    
    prev_stage = None
    try:
        for episode in range(start_episode, Config.NUM_EPISODES):
            # Get curriculum settings
            curr_config = curriculum.get_config(episode)
            
            # Update trainer settings
            trainer.set_entropy_coef(curr_config['entropy_coef'])
            max_steps = curr_config['max_steps']
            exploration_noise = curr_config['exploration_noise']
            
            # Print stage changes
            if curriculum.stage != prev_stage:
                print(f"\n{'=' * 70}")
                print(f"{curr_config['description']}")
                print(f"Max Steps: {max_steps}")
                print(f"Entropy: {curr_config['entropy_coef']:.3f}")
                print(f"Exploration Noise: {exploration_noise:.3f}")
                print(f"{'=' * 70}\n")
                prev_stage = curriculum.stage
            
            # Reset environment and all helpers
            obs = env.reset()
            reward_shaper.reset()
            turn_detector.reset()
            warm_start.reset()
            action_smoother.reset()
            # Reset lap counters/timers per episode
            last_lap_count = 0
            last_lap_step = 0
            prev_progress_ep = float(obs.get('progress', 0.0))
            
            # Video recording for successful laps
            recording_frames = []
            recording = False
            
            # Episode tracking
            episode_reward = 0
            episode_shaped_reward = 0
            episode_speeds = []
            episode_actions = []
            episode_turn_types = []
            episode_best_progress = 0.0
            exploit_steps_remaining = 0
            base_entropy_coef = curr_config['entropy_coef']
            recovery_window = 0
            stability_counter = 0
            speed_mode_steps = 0
            printed_repetitive = False
            episode_lap_recorded = False
            
            trajectories = {
                'states_img': [],
                'states_sensor': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            }
            
            for step in range(max_steps):
                # Prepare observation
                image = torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0)
                sensors = torch.FloatTensor([
                    obs['speed'], 
                    obs['cte'], 
                    obs['angle'], 
                    obs['progress']
                ])
                
                # Record frames for potential successful lap
                if recording and len(recording_frames) < 1000:  
                    recording_frames.append(obs['image'].copy())
                
                
                # Track metrics
                episode_speeds.append(obs['speed'])
                
                # TURN DETECTION 
                turn_detector.update(obs['cte'], obs['angle'], obs['progress'])
                turn_type, turn_direction, turn_confidence = turn_detector.detect_turn_type()
                steering_multiplier = turn_detector.get_steering_multiplier()
                episode_turn_types.append(turn_type)
                # Turn-aware speed management
                should_slow, target_speed = turn_detector.should_slow_down()
                # Cap throttle by turn type (original, less restrictive caps)
                if turn_type == 'hairpin':
                    throttle_cap = 0.35  
                elif turn_type == 'sharp':
                    throttle_cap = 0.45  
                elif turn_type == 'medium':
                    throttle_cap = 0.60  
                else:
                    throttle_cap = 0.8
                if should_slow:
                    if target_speed <= 2.5:
                        throttle_cap = min(throttle_cap, 0.35)
                    elif target_speed <= 3.5:
                        throttle_cap = min(throttle_cap, 0.45)
                    elif target_speed <= 4.8:
                        throttle_cap = min(throttle_cap, 0.55)
                    else:
                        throttle_cap = min(throttle_cap, 0.70)
                    throttle_cap = max(0.0, throttle_cap - 0.02)

                # TWO-PHASE SPEED STRATEGY 
                # Phase detection: track stability and progress; when stable, enter speed mode window
                cte_abs_live = abs(obs.get('cte', 0.0))
                angle_abs_live = abs(obs.get('angle', 0.0))
                if (cte_abs_live < 0.6) and (angle_abs_live < 0.25):
                    stability_counter = min(stability_counter + 1, 200)
                else:
                    stability_counter = max(stability_counter - 1, 0)
                if (stability_counter >= 20 or reward_shaper.max_progress > 0.30) and speed_mode_steps == 0:
                    speed_mode_steps = 80

                # Apply speed mode: raise throttle caps when safe (not hairpin) and boost min throttle
                if speed_mode_steps > 0:
                    if turn_type in ['straight', 'gentle', 'medium']:
                        throttle_cap = min(0.95, throttle_cap + 0.15)
                    elif turn_type == 'sharp':
                        throttle_cap = min(0.60, throttle_cap + 0.08)
                    try:
                        env.set_min_throttle(0.25)
                    except Exception:
                        pass
                    speed_mode_steps -= 1
                # Dynamically reduce minimum throttle to allow slowing down
                try:
                    if should_slow:
                        env.set_min_throttle(0.08)  
                    else:
                        env.set_min_throttle(0.15)  
                except Exception:
                    pass

                # LEARNED SPEED PROFILE BY PROGRESS BIN
                try:
                    prog = float(obs.get('progress', 0.0))
                except Exception:
                    prog = 0.0
                bin_idx = int(max(0, min(PROFILE_BINS - 1, int(prog * PROFILE_BINS))))
                # After first lap(s), prefer learned throttle for this segment when stable
                if (last_lap_count > 0 or success_count > 0):
                    if abs(obs.get('cte', 0.0)) < 0.9 and abs(obs.get('angle', 0.0)) < 0.25:
                        learned_thr = float(speed_profile[bin_idx])
                        if learned_thr > 0.0:
                            throttle_cap = max(throttle_cap, min(1.0, learned_thr + 0.10))
                
                # Refresh exploitation window based on improved progress
                if reward_shaper.max_progress > episode_best_progress + 0.01:
                    episode_best_progress = reward_shaper.max_progress
                    exploit_steps_remaining = 80

                # ACTION SELECTION
                warm_action = warm_start.get_action(step, obs.get('cte', 0.0))
                
                if warm_action is not None:
                    # Use warm start action
                    action_np = warm_action
                    action = torch.FloatTensor(action_np).unsqueeze(0).to(device)
                    
                    # Get value for trajectory
                    with torch.no_grad():
                        _, _, value = model(image.to(device), sensors.unsqueeze(0).to(device))
                    
                    log_prob = torch.zeros(1, 1).to(device)
                    
                else:
                    # Use policy
                    with torch.no_grad():
                        action_mean, action_std, value = model(
                            image.to(device), 
                            sensors.unsqueeze(0).to(device)
                        )
                        
                        # Progress-gated exploitation and turn-aware confidence gating
                        if exploit_steps_remaining > 0:
                            trainer.set_entropy_coef(max(0.004, base_entropy_coef * 0.2))
                            effective_noise = exploration_noise * 0.2
                            exploit_steps_remaining -= 1
                        else:
                            trainer.set_entropy_coef(base_entropy_coef)
                            effective_noise = exploration_noise
                        if (turn_type in ['sharp', 'hairpin']) and (turn_confidence > 0.7):
                            trainer.set_entropy_coef(max(0.004, base_entropy_coef * 0.15))
                            effective_noise = min(effective_noise, exploration_noise * 0.15)

                        # Add exploration noise and clamp std
                        if effective_noise > 0:
                            action_std = action_std + effective_noise
                        action_std = torch.clamp(action_std, min=0.05, max=0.6)
                        
                        dist = torch.distributions.Normal(action_mean, action_std)
                        # Deterministic action during exploitation window, else sample
                        if exploit_steps_remaining > 0:
                            action = action_mean
                        else:
                            action = dist.sample()
                        action = torch.clamp(action, -1.0, 1.0)
                        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                    
                    action_np = action.squeeze().cpu().numpy()
                    
                    # Apply steering multiplier for sharp turns (reduced in TurnDetector)
                    action_np[0] = action_np[0] * steering_multiplier
                    action_np[0] = np.clip(action_np[0], -0.98, 0.98)

                    # Post-turn steering bias to avoid going too straight after a turn
                    if hasattr(turn_detector, 'is_in_cooldown') and turn_detector.is_in_cooldown():
                        last_dir = getattr(turn_detector, 'get_last_turn_direction', lambda: 'none')()
                        bias = 0.09 if last_dir == 'right' else (-0.09 if last_dir == 'left' else 0.0)
                        if bias != 0.0:
                            action_np[0] = np.clip(action_np[0] + bias, -0.98, 0.98)
                    
                    # Throttle capping on turns
                    if should_slow:
                        action_np[1] = float(min(action_np[1], throttle_cap))
                    
                    # Always apply smoothing with dynamic rate limits
                    # During cooldown, allow quicker steering but slower throttle changes
                    if hasattr(turn_detector, 'is_in_cooldown') and turn_detector.is_in_cooldown():
                        action_smoother.set_rate_limits(max_steer_rate=0.38, max_throttle_rate=0.08)
                    else:
                        if turn_type == 'hairpin':
                            action_smoother.set_rate_limits(max_steer_rate=0.45, max_throttle_rate=0.12)
                        elif turn_type == 'sharp':
                            action_smoother.set_rate_limits(max_steer_rate=0.40, max_throttle_rate=0.12)
                        elif turn_type == 'medium':
                            action_smoother.set_rate_limits(max_steer_rate=0.30, max_throttle_rate=0.10)
                        else:
                            action_smoother.set_rate_limits(max_steer_rate=0.22, max_throttle_rate=0.08)

                    # In speed mode, allow slightly faster throttle ramp to reach higher speeds
                    if speed_mode_steps > 0 and turn_type in ['straight', 'gentle', 'medium']:
                        action_smoother.set_rate_limits(max_steer_rate=0.30, max_throttle_rate=0.15)

                    # If far off the center line, relax steering rate limits to recover faster
                    cte_abs = abs(obs.get('cte', 0.0))
                    if cte_abs > 1.6:
                        action_smoother.set_rate_limits(max_steer_rate=0.75, max_throttle_rate=0.10)
                    elif cte_abs > 1.2:
                        action_smoother.set_rate_limits(max_steer_rate=0.60, max_throttle_rate=0.10)
                    action_np = action_smoother.smooth(action_np)
                    
                    # Apply learned minimum throttle for this segment when confident
                    if (last_lap_count > 0 or success_count > 0):
                        if abs(obs.get('cte', 0.0)) < 0.9 and abs(obs.get('angle', 0.0)) < 0.25:
                            learned_thr = float(speed_profile[bin_idx])
                            if learned_thr > 0.0:
                                action_np[1] = float(max(action_np[1], min(1.0, learned_thr)))
                
                # Track action diversity
                episode_actions.append(action_np.copy())
                
                # Simple recovery: if stalled or far off center at low speed, steer back to center with gentle throttle
                try:
                    if (reward_shaper.no_progress_counter > 30) or (abs(obs['cte']) > 2.5 and obs['speed'] < 1.0):
                        steer_dir = 0.0 if obs['cte'] == 0 else -np.sign(obs['cte'])
                        action_np = np.array([steer_dir * 0.6, 0.2], dtype=np.float32)
                        # allow slowing during recovery
                        try:
                            env.set_min_throttle(0.05)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Safety controller: strong recentering and throttle cap when off-track
                try:
                    cte_abs = abs(obs.get('cte', 0.0))
                    speed = float(obs.get('speed', 0.0))
                    if cte_abs > 1.8:
                        steer_dir = 0.0 if obs.get('cte', 0.0) == 0 else -np.sign(obs['cte'])
                        action_np[0] = float(np.clip(steer_dir * 0.85, -0.98, 0.98))
                        action_np[1] = 0.02
                        try:
                            env.set_min_throttle(0.0)
                        except Exception:
                            pass
                    elif cte_abs > 1.0:
                        steer_dir = 0.0 if obs.get('cte', 0.0) == 0 else -np.sign(obs['cte'])
                        recover_steer = max(0.35, min(0.75, cte_abs * 0.45))
                        action_np[0] = float(np.clip(steer_dir * recover_steer, -0.98, 0.98))
                        action_np[1] = float(min(action_np[1], 0.06))
                        try:
                            env.set_min_throttle(0.05)
                        except Exception:
                            pass
                    prog_delta = float(obs.get('progress', 0.0)) - float(getattr(reward_shaper, 'prev_progress', 0.0))
                    if (cte_abs > 1.0) and (prog_delta < 0.002):
                        action_np[1] = float(min(action_np[1], 0.06))

                    # Keep-alive throttle boost: if on-track but not progressing (likely too slow through back-to-back turns)
                    if (cte_abs < 0.8) and (reward_shaper.no_progress_counter > 20):
                        action_np[1] = float(max(action_np[1], 0.18))
                        try:
                            env.set_min_throttle(0.10)
                        except Exception:
                            pass
                    elif (cte_abs < 1.0) and (turn_type in ['sharp', 'hairpin']) and (turn_confidence > 0.6) and (reward_shaper.no_progress_counter > 10):
                        action_np[1] = float(max(action_np[1], 0.14))
                        try:
                            env.set_min_throttle(0.08)
                        except Exception:
                            pass

                    # Speed-based keep-alive: if on-track but too slow, nudge throttle up
                    if (cte_abs < 0.8) and (speed < 0.8):
                        action_np[1] = float(max(action_np[1], 0.20))
                        try:
                            env.set_min_throttle(0.10)
                        except Exception:
                            pass
                    elif (cte_abs < 1.0) and (turn_type in ['sharp', 'hairpin']) and (speed < 1.2):
                        action_np[1] = float(max(action_np[1], 0.16))
                        try:
                            env.set_min_throttle(0.08)
                        except Exception:
                            pass
                    elif speed < 0.4:
                        action_np[1] = float(max(action_np[1], 0.12))

                    # Post-recovery momentum window: if we just re-centered after being wide
                    try:
                        prev_cte_abs = abs(getattr(reward_shaper, 'prev_cte', 0.0))
                    except Exception:
                        prev_cte_abs = cte_abs
                    if (recovery_window == 0) and (prev_cte_abs > 1.0) and (cte_abs < 0.8):
                        recovery_window = 25
                    if (recovery_window > 0) and (cte_abs < 0.8):
                        action_np[1] = float(max(action_np[1], 0.16))
                        try:
                            env.set_min_throttle(0.10)
                        except Exception:
                            pass
                        recovery_window -= 1
                except Exception:
                    pass

                next_obs, env_reward, done, info = env.step(action_np)
                
                # Calculate shaped reward with turn awareness
                shaped_reward = reward_shaper.calculate_reward(
                    obs, done, info, turn_type, turn_direction
                )
                
                # Store trajectory
                trajectories['states_img'].append(image)
                trajectories['states_sensor'].append(sensors)
                trajectories['actions'].append(action)
                trajectories['rewards'].append(shaped_reward)
                trajectories['values'].append(value.squeeze())
                trajectories['log_probs'].append(log_prob.squeeze())
                trajectories['dones'].append(float(done))
                
                episode_reward += env_reward
                episode_shaped_reward += shaped_reward
                obs = next_obs

                # Update learned throttle profile on safe, stable steps
                try:
                    if (not info.get('hit', False)) and abs(obs.get('cte', 0.0)) < 0.9 and abs(obs.get('angle', 0.0)) < 0.3:
                        thr_exec = float(action_np[1])
                        if thr_exec > speed_profile[bin_idx]:
                            speed_profile[bin_idx] = thr_exec
                        profile_counts[bin_idx] += 1
                except Exception:
                    pass
                
                # Early termination if stuck
                if step > 150 and reward_shaper.max_progress < 0.03:
                    if episode % 100 == 0:
                        print(f"Episode {episode + 1}: Stuck at start (progress: {reward_shaper.max_progress:.3f})")
                    stuck_count += 1
                    done = True
                
                # Repetitive behavior detection
                if step > 50:
                    recent_actions = episode_actions[-50:] if len(episode_actions) >= 50 else episode_actions
                    if len(recent_actions) > 10:
                        steering_variance = np.var([a[0] for a in recent_actions])
                        if steering_variance < 0.01:
                            if (not printed_repetitive) and Config.VERBOSE:
                                print(f"Episode {episode + 1}: Repetitive behavior")
                                printed_repetitive = True
                            if step > 200:
                                try:
                                    env.set_min_throttle(0.10)
                                except Exception:
                                    pass
                                recovery_window = max(recovery_window, 20)
                
                # Mid-episode lap detection (env flag OR progress wrap-around)
                try:
                    lap_ct = int(info.get('lap_count', 0))
                except Exception:
                    lap_ct = 0
                new_lap_time = info.get('last_lap_time', None)
                curr_progress = float(obs.get('progress', 0.0))
                
                # DEBUG: Print lap detection values every 100 steps
                if step % 100 == 0:
                    print(f"DEBUG EP{episode+1} STEP{step}: lap_count={lap_ct}, progress={curr_progress:.3f}, recording={recording}, frames={len(recording_frames)}")
                
                # Start recording at the beginning of a potential lap - very aggressive conditions
                if curr_progress < 0.3 and not recording and not info.get('hit', False):
                    recording = True
                    recording_frames = []
                    print(f"DEBUG: Started recording at progress {curr_progress:.3f}")
                # Also start recording if we have any speed and are not recording
                elif not recording and obs.get('speed', 0) > 1.0 and not info.get('hit', False):
                    recording = True
                    recording_frames = []
                    print(f"DEBUG: Started recording at speed {obs.get('speed', 0):.1f} with progress {curr_progress:.3f}")
                # Start recording immediately at episode start if not recording
                elif step == 0 and not recording:
                    recording = True
                    recording_frames = []
                    print(f"DEBUG: Started recording at episode start")
                progress_wrapped_mid = (prev_progress_ep >= 0.85 and curr_progress < 0.20 and not info.get('hit', False))
                # More sensitive lap detection - accept smaller progress increments
                lap_count_increment = (lap_ct > last_lap_count) 
                progress_jump = (curr_progress > 0.1 and prev_progress_ep < 0.05)  # Jump from near 0 to measurable progress
                
                lap_detected = lap_count_increment or progress_wrapped_mid or progress_jump
                
                # DEBUG: Print lap detection details
                if lap_detected:
                    print(f"DEBUG LAP DETECTED: lap_ct={lap_ct}, last_lap_count={last_lap_count}, progress_wrapped={progress_wrapped_mid}, progress_jump={progress_jump}")
                
                if lap_detected:
                    # Determine lap time
                    if isinstance(new_lap_time, (int, float)) and new_lap_time and np.isfinite(new_lap_time) and new_lap_time > 0:
                        lt = float(new_lap_time)
                    else:
                        lt = max(0.0, (step - last_lap_step) * 0.05)
                    last_lap_step = step
                    last_lap_count = max(last_lap_count + 1, lap_ct)
                    lap_times.append(lt)
                    episode_lap_recorded = True
                    success_count += 1
                    total_laps_completed += 1
                    # Count mid-episode lap as successful episode
                    episode_successes.append(1)
                    # Save recording only if this lap is the fastest so far; save a single GIF (overwrite)
                    if recording and len(recording_frames) > 5 and (lt > 0) and (lt < best_lap_time):
                        print(f"DEBUG: New best lap {lt:.2f}s with {len(recording_frames)} frames — saving best_lap.gif")
                        try:
                            # Convert frames to uint8 if needed
                            converted_frames = []
                            for frame in recording_frames:
                                if frame.dtype != np.uint8:
                                    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                                    converted_frames.append(frame_uint8)
                                else:
                                    converted_frames.append(frame)
                            # Upscale frames for better visual quality
                            scale = 3
                            h, w = converted_frames[0].shape[:2]
                            target_size = (w * scale, h * scale)
                            up_frames = [cv2.resize(f, target_size, interpolation=cv2.INTER_CUBIC) for f in converted_frames]
                            # Save GIF and MP4 for the fastest lap (overwrite)
                            gif_path = os.path.join(Config.RESULTS_DIR, "best_lap.gif")
                            imageio.mimsave(gif_path, up_frames, fps=40)
                            mp4_path = os.path.join(Config.RESULTS_DIR, "best_lap.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(mp4_path, fourcc, 40, target_size)
                            for f in up_frames:
                                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                            out.release()
                            print(f"   📹 Best lap saved: {gif_path} and {mp4_path}")
                        except Exception as e:
                            print(f"   Warning: Could not save best lap GIF: {e}")
                    recording = False
                    recording_frames = []
                    # Reset per-lap milestones so thresholds apply within each lap
                    try:
                        reward_shaper.reset_lap_milestones()
                    except Exception:
                        pass
                    if lt < best_lap_time and lt > 0:
                        best_lap_time = lt
                        trainer.save_checkpoint(best_model_path)
                        print(f"   🏁 LAP COMPLETE mid-episode: {lt:.2f}s - Saved best model")
                prev_progress_ep = curr_progress

                if done:
                    final_progress = float(obs.get('progress', reward_shaper.max_progress))
                    success_flag = False
                    # Only count a success if we have an explicit lap boundary
                    env_lap_flag = bool(info.get('lap', False)) or bool(info.get('completed_lap', False)) or bool(info.get('lap_complete', False))
                    # Detect wrap-around at the end
                    try:
                        prev_prog = float(getattr(reward_shaper, 'prev_progress', 0.0))
                    except Exception:
                        prev_prog = final_progress
                    wrap_flag = (prev_prog >= 0.85) and (final_progress < 0.20) and (not info.get('hit', False))
                    # Also accept env lap_count increment right before done
                    try:
                        lap_ct_end = int(info.get('lap_count', 0))
                    except Exception:
                        lap_ct_end = last_lap_count
                    counter_flag = (lap_ct_end > last_lap_count)
                    
                    # Additional progress-based success detection
                    progress_success = (final_progress > 0.95 and not info.get('hit', False))
                    
                    if env_lap_flag or wrap_flag or counter_flag or progress_success:
                        success_flag = True
                    if success_flag and not info.get('hit', False):
                        success_count += 1
                        if not episode_lap_recorded:
                            lap_time = max(0.0, (step - last_lap_step) * 0.05)
                            lap_times.append(lap_time)
                            episode_lap_recorded = True
                            total_laps_completed += 1  # Increment total laps here too
                            print(f"Episode {episode + 1}: LAP COMPLETE! Time: {lap_time:.2f}s, Progress: {final_progress:.3f}")
                            try:
                                reward_shaper.reset_lap_milestones()
                            except Exception:
                                pass
                        else:
                            # Lap already recorded mid-episode, just acknowledge
                            print(f"Episode {episode + 1}: LAP COMPLETE! Time: {lap_time:.2f}s, Progress: {final_progress:.3f}")
                        if lap_time < best_lap_time:
                            best_lap_time = lap_time
                            trainer.save_checkpoint(best_model_path)
                            print(f"Best lap improved to {best_lap_time:.2f}s - Model saved!")
                    elif info.get('hit', False):
                        collision_count += 1
                        print(f"Episode {episode + 1}: Collision at progress {final_progress:.3f}")
                        # Reset recording on collision
                        recording = False
                        recording_frames = []
                    episode_successes.append(1 if (success_flag and not info.get('hit', False)) else 0)
                    break
            
            # Reset recording at episode end
            recording = False
            recording_frames = []
            
            # Calculate statistics
            episode_actions = np.array(episode_actions)
            steering_std = episode_actions[:, 0].std() if len(episode_actions) > 0 else 0
            throttle_std = episode_actions[:, 1].std() if len(episode_actions) > 0 else 0
            
            # Count turn types encountered
            from collections import Counter
            turn_counts = Counter(episode_turn_types)
            for turn_type in turn_counts:
                turn_type_stats[turn_type] = turn_type_stats.get(turn_type, 0) + turn_counts[turn_type]
            
            # Store next value for GAE
            with torch.no_grad():
                image = torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0)
                sensors = torch.FloatTensor([obs['speed'], obs['cte'], obs['angle'], obs['progress']])
                _, _, next_value = model(image.to(device), sensors.unsqueeze(0).to(device))
                trajectories['next_value'] = next_value.item()
            
            # Update policy
            if (episode + 1) % Config.UPDATE_FREQUENCY == 0 and len(trajectories['rewards']) > 0:
                losses = trainer.update(trajectories)
                
                if Config.VERBOSE and (episode + 1) % Config.LOG_FREQUENCY == 0:
                    print(f"\nPolicy Update:")
                    print(f"Policy Loss: {losses['policy_loss']:.4f}")
                    print(f"Value Loss: {losses['value_loss']:.4f}")
                    print(f"Entropy: {losses['entropy']:.4f}")
            
            # Store metrics
            episode_rewards.append(episode_shaped_reward)
            episode_lengths.append(step + 1)
            episode_progress.append(reward_shaper.max_progress)
            
            # No live plotting updates
            
            # Save best model by progress (separate)
            if reward_shaper.max_progress > best_progress:
                best_progress = reward_shaper.max_progress
                trainer.save_checkpoint(best_progress_model_path)
                if best_progress > 0.1:
                    print(f"New best progress: {best_progress:.3f} - Progress model saved!")
            
            # Detailed logging
            if Config.VERBOSE and (episode + 1) % Config.LOG_FREQUENCY == 0:
                avg_reward = np.mean(episode_rewards[-Config.LOG_FREQUENCY:])
                avg_length = np.mean(episode_lengths[-Config.LOG_FREQUENCY:])
                avg_progress = np.mean(episode_progress[-Config.LOG_FREQUENCY:])
                avg_speed = np.mean(episode_speeds) if episode_speeds else 0
                success_rate = success_count / (episode + 1 - start_episode) * 100
                
                print(f"\n{'=' * 70}")
                print(f"Episode {episode + 1}/{Config.NUM_EPISODES} - {curr_config['description']}")
                print(f"{'=' * 70}")
                print(f"   Reward (avg last {Config.LOG_FREQUENCY}): {avg_reward:.2f}")
                print(f"   Episode Length: {avg_length:.1f} steps")
                print(f"   Progress: {reward_shaper.max_progress:.3f} (avg: {avg_progress:.3f})")
                print(f"   Best Progress Ever: {best_progress:.3f}")
                print(f"   Average Speed: {avg_speed:.2f}")
                print(f"   Success Rate: {success_rate:.2f}% ({success_count} laps)")
                print(f"   Collisions: {collision_count}")
                print(f"   Stuck Episodes: {stuck_count}")
                
                # Turn type statistics
                print(f"\n Turn Type Distribution (this episode):")
                total_steps = sum(turn_counts.values())
                for ttype in ['straight', 'gentle', 'medium', 'sharp', 'hairpin']:
                    count = turn_counts.get(ttype, 0)
                    pct = (count / total_steps * 100) if total_steps > 0 else 0
                    print(f"      {ttype.capitalize()}: {count} ({pct:.1f}%)")
                
                # Action diversity
                print(f"\n   Exploration Metrics:")
                print(f"      Steering Std: {steering_std:.3f}")
                print(f"      Throttle Std: {throttle_std:.3f}")
                if steering_std < 0.05:
                    print(f"      WARNING: Low steering diversity!")
                else:
                    print(f"       Good exploration")
                
                if lap_times:
                    print(f"\n   Lap Times:")
                    print(f"      Best: {min(lap_times):.2f}s")
                    print(f"      Average: {np.mean(lap_times):.2f}s")
                
                print(f"{'=' * 70}\n")
            
            # Save checkpoint
            if (episode + 1) % Config.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_episode_{episode + 1}.pt")
                trainer.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

            # Persist metrics each episode to avoid blank graphs on early exit
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_progress': episode_progress,
                'lap_times': lap_times,
                'success_count': success_count,
                'success_rate': (sum(episode_successes) / max(1, len(episode_successes))) * 100,
                'best_progress': best_progress,
                'stuck_count': stuck_count,
                'collision_count': collision_count,
                'turn_type_stats': turn_type_stats,
                'episode_successes': episode_successes,
                'speed_profile': speed_profile.tolist(),
                'total_laps_completed': total_laps_completed
            }
            metrics_path = os.path.join(Config.RESULTS_DIR, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
    
    except KeyboardInterrupt:
        print("\n\n  Training interrupted!")
        interrupted_path = os.path.join(Config.CHECKPOINT_DIR, f"model_interrupted_ep{episode}.pt")
        trainer.save_checkpoint(interrupted_path)
        print(f" Progress saved to: {interrupted_path}")
    
    finally:
        # Cleanup
        if 'plotter' in locals() and plotter:
            plotter.close()
        
        # Final save
        final_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_final.pt")
        trainer.save_checkpoint(final_model_path)
        print(f"\n Final model saved: {final_model_path}")
        
        # Save metrics
        metrics = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_progress': episode_progress,
            'lap_times': lap_times,
            'success_count': success_count,
            'success_rate': (sum(episode_successes) / max(1, len(episode_successes))) * 100,
            'best_progress': best_progress,
            'stuck_count': stuck_count,
            'collision_count': collision_count,
            'turn_type_stats': turn_type_stats,
            'episode_successes': episode_successes,
            'total_laps_completed': total_laps_completed
        }
        
        metrics_path = os.path.join(Config.RESULTS_DIR, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f" Metrics saved: {metrics_path}")
        
        # Save training plot using matplotlib
        try:
            plot_path = os.path.join(Config.RESULTS_DIR, "training_progress.png")
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            # Episode rewards
            axes[0].plot(episode_rewards, label='Shaped Reward', color='tab:blue')
            axes[0].set_title('Episode Rewards')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True, alpha=0.3)
            # Episode lengths
            axes[1].plot(episode_lengths, label='Episode Length', color='tab:orange')
            axes[1].set_title('Episode Lengths (steps)')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Steps')
            axes[1].grid(True, alpha=0.3)
            # Episode progress with moving average
            axes[2].plot(episode_progress, label='Progress', color='tab:green')
            if len(episode_progress) >= 10:
                ma = np.convolve(np.array(episode_progress), np.ones(10) / 10, mode='valid')
                axes[2].plot(range(9, 9 + len(ma)), ma, label='MA(10)', color='tab:red')
            axes[2].set_title('Episode Progress (with MA)')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Progress')
            axes[2].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Training plot saved: {plot_path}")
        except Exception as _e:
            pass

        # Deterministic evaluation run for fast-lap GIF (ensure full, collision-free lap)
        try:
            eval_env = env
            model.eval()
            with torch.no_grad():
                obs = eval_env.reset()
                speeds = []
                times = []
                frames = []
                t = 0.0
                eval_steps = 0
                # Load best-performing model if available
                best_path = best_model_path if os.path.exists(best_model_path) else (best_progress_model_path if os.path.exists(best_progress_model_path) else None)
                if best_path:
                    checkpoint = torch.load(best_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])

                # State machine to capture exactly one full, collision-free lap
                recording = False
                start_lap_count = None
                base_lap_count = 0
                prev_progress = float(obs.get('progress', 0.0))
                collided = False
                best_frames = None

                # Allow up to 2x max steps to find a lap start and complete one
                while eval_steps < Config.MAX_STEPS_PER_EPISODE * 2:
                    image = torch.FloatTensor(obs['image']).permute(2, 0, 1).unsqueeze(0)
                    sensors = torch.FloatTensor([obs['speed'], obs['cte'], obs['angle'], obs['progress']])
                    action_mean, action_std, _ = model(image.to(device), sensors.unsqueeze(0).to(device))
                    action = torch.clamp(action_mean, -1.0, 1.0).squeeze().cpu().numpy()

                    # Slightly aggressive throttle for eval to keep momentum
                    a = action.copy()
                    cte_abs = abs(float(obs.get('cte', 0.0)))
                    if cte_abs < 0.8:
                        a[1] = float(min(1.0, max(a[1], 0.9)))
                    else:
                        a[1] = float(min(0.6, max(a[1], 0.2)))

                    next_obs, _, done, info = eval_env.step(a)

                    # Lap boundary signals
                    try:
                        lap_count = int(info.get('lap_count', 0))
                    except Exception:
                        lap_count = 0
                    last_lap_time = info.get('last_lap_time', 0)
                    progress = float(next_obs.get('progress', 0.0)) if isinstance(next_obs, dict) else 0.0
                    progress_wrapped = (prev_progress >= 0.90 and progress < 0.15)
                    hit_flag = info.get('hit', 'none')
                    if hit_flag and hit_flag != 'none':
                        collided = True

                    # Start recording only once we cross start/finish
                    if not recording:
                        if base_lap_count == 0:
                            base_lap_count = lap_count
                        if (lap_count > base_lap_count) or progress_wrapped:
                            recording = True
                            start_lap_count = lap_count if lap_count > 0 else (base_lap_count + 1)
                            frames = []
                            speeds = []
                            times = []
                            t = 0.0
                            collided = False
                    else:
                        # Record frame from current obs image (float 0..1) as uint8
                        try:
                            img = obs['image']
                            if img is not None:
                                if img.dtype != np.uint8:
                                    frame = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                                else:
                                    frame = img
                                # Overlay a simple car marker at bottom-center so the car is visible
                                try:
                                    h, w = frame.shape[0], frame.shape[1]
                                    cw, ch = max(6, w // 30), max(6, h // 30)
                                    x1 = w // 2 - cw // 2
                                    y1 = h - ch - 2
                                    x2 = x1 + cw
                                    y2 = y1 + ch
                                    frame[y1:y2, x1:x2] = [255, 30, 30]
                                except Exception:
                                    pass
                                frames.append(frame)
                        except Exception:
                            pass
                        speeds.append(float(obs.get('speed', 0.0)))
                        times.append(t)
                        t += 0.05

                    obs = next_obs
                    prev_progress = progress
                    eval_steps += 1

                    # Stop when the recorded lap completes
                    if recording:
                        if (lap_count > (start_lap_count if start_lap_count is not None else 0)) or progress_wrapped:
                            # Completed a lap window
                            if (not collided) and len(frames) > 10:
                                best_frames = list(frames)
                                break
                            else:
                                # Discard and search for another clean lap within remaining steps
                                recording = False
                                frames = []
                                speeds = []
                                times = []
                                t = 0.0
                                collided = False
                    if done and not recording:
                        # If episode ended before we started recording, reset once and continue
                        try:
                            obs = eval_env.reset()
                            prev_progress = float(obs.get('progress', 0.0))
                            done = False
                            continue
                        except Exception:
                            break

                # Save only a GIF/MP4 if we captured frames for a full, collision-free lap (overwrite best_lap files)
                if best_frames:
                    try:
                        # Convert and upscale frames for high-res save
                        converted = []
                        for frame in best_frames:
                            if frame.dtype != np.uint8:
                                f = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                            else:
                                f = frame
                            converted.append(f)
                        scale = 3
                        h, w = converted[0].shape[:2]
                        target_size = (w * scale, h * scale)
                        up_frames = [cv2.resize(f, target_size, interpolation=cv2.INTER_CUBIC) for f in converted]
                        gif_path = os.path.join(Config.RESULTS_DIR, "best_lap.gif")
                        imageio.mimsave(gif_path, up_frames, fps=40)
                        mp4_path = os.path.join(Config.RESULTS_DIR, "best_lap.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(mp4_path, fourcc, 40, target_size)
                        for f in up_frames:
                            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                        out.release()
                        print(f"Best-lap saved: {gif_path} and {mp4_path}")
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Stop screen recorder before final summary
        try:
            if screen_recorder:
                screen_recorder.stop()
                print(" Screen recording stopped.")
        except Exception:
            pass

        # Final summary
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total Episodes: {len(episode_rewards)}")
        try:
            succ_rate = (sum(episode_successes) / max(1, len(episode_successes))) * 100
        except Exception:
            succ_rate = 0.0
        print(f"Successful Laps (episodes): {sum(episode_successes)} ({succ_rate:.2f}%)")
        try:
            print(f"Total Laps Detected (mid-episode): {total_laps_completed}")
        except Exception:
            pass
        print(f"Best Progress: {best_progress:.3f}")
        print(f"Stuck Episodes: {stuck_count}")
        print(f"Collisions: {collision_count}")
        if lap_times:
            print(f"Best Lap Time: {min(lap_times):.2f}s")
        print(f"\n Best model: {best_model_path}")
        print("=" * 70)
        
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO racing agent')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  PPO Racing Agent - Cleaned")
    print("=" * 70)
    print("\n Features:")
    print("  Curriculum learning")
    print("  Turn detection with throttle management")
    print("  Turn-aware reward shaping")
    print("  Warm start & action smoothing")
    print("  Stuck detection & recovery")
    print("  Best model tracking & training plot")
    print("=" * 70 + "\n")
    
    train(
        resume_from=args.resume
    )