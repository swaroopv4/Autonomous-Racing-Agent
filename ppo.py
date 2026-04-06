import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config.config import Config

class PPOTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Hyperparameters
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        self.clip_epsilon = Config.CLIP_EPSILON
        self.epochs = Config.EPOCHS_PER_UPDATE
        self.batch_size = Config.BATCH_SIZE
        self.entropy_coef = Config.ENTROPY_COEF
        self.value_loss_coef = Config.VALUE_LOSS_COEF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, trajectories):
        # Extract trajectory data
        states_img = torch.cat(trajectories['states_img']).to(self.device)
        states_sensor = torch.stack(trajectories['states_sensor']).to(self.device)
        actions = torch.cat(trajectories['actions']).to(self.device)
        old_log_probs = torch.stack(trajectories['log_probs']).to(self.device)
        
        # Compute advantages and returns
        rewards = trajectories['rewards']
        values = [v.item() for v in trajectories['values']]
        dones = trajectories['dones']
        next_value = trajectories.get('next_value', 0)
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        dataset_size = states_img.size(0)
        # Safeguard against empty trajectories
        if dataset_size == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        
        for epoch in range(self.epochs):
            early_stop = False
            # Generate random indices for mini-batches
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                batch_img = states_img[batch_indices]
                batch_sensor = states_sensor[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy predictions
                action_mean, action_std, values = self.model(batch_img, batch_sensor)
                dist = torch.distributions.Normal(action_mean, action_std)
                
                # Calculate new log probs
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Calculate ratio and clipped surrogate
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(-1)
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1

                # Approximate KL divergence for early stopping (ensure at least one update occurred)
                approx_kl = (batch_old_log_probs - new_log_probs).mean()
                if approx_kl.item() > Config.TARGET_KL:
                    early_stop = True
                    break
            if early_stop:
                break
        
        # Return average losses
        if update_count == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        else:
            return {
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy': total_entropy / update_count
            }
    
    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])