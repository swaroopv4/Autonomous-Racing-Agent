from config.config import Config

class Curriculum:    
    def __init__(self):
        self.stage = 0
        self.episode_count = 0
        
    def get_config(self, episode):
        self.episode_count = episode
        total_eps = Config.NUM_EPISODES
        s1 = int(0.40 * total_eps)
        s2 = int(0.75 * total_eps)
        s3 = int(0.95 * total_eps)
        
        # Stage 1: VERY AGGRESSIVE exploration
        if episode < s1:
            self.stage = 1
            return {
                'max_steps': Config.MAX_STEPS_PER_EPISODE, 
                'entropy_coef': max(0.04, Config.ENTROPY_COEF * 1.2),  
                'exploration_noise': 0.5,  
                'description': '🎯 Stage 1: Aggressive turn exploration'
            }
        
        # Stage 2: Learn to complete turns
        elif episode < s2:
            self.stage = 2
            return {
                'max_steps': Config.MAX_STEPS_PER_EPISODE,
                'entropy_coef': max(0.03, Config.ENTROPY_COEF * 0.7),  
                'exploration_noise': 0.12,
                'description': '🏁 Stage 2: Mastering turns'
            }
        
        # Stage 3: Optimize speed through turns
        elif episode < s3:
            self.stage = 3
            return {
                'max_steps': Config.MAX_STEPS_PER_EPISODE,
                'entropy_coef': max(0.015, Config.ENTROPY_COEF * 0.4), 
                'exploration_noise': 0.05,
                'description': '🚀 Stage 3: Speed optimization'
            }
        
        # Stage 4: Fine-tuning
        else:
            self.stage = 4
            return {
                'max_steps': Config.MAX_STEPS_PER_EPISODE,
                'entropy_coef': max(0.005, Config.ENTROPY_COEF * 0.2),  
                'exploration_noise': 0.01,
                'description': '⚡ Stage 4: Fine-tuning'
            }