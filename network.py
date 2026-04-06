import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActorCritic(nn.Module):
    def __init__(self, action_dim=2):
        super(CNNActorCritic, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 11 * 16  
        
        # Sensor processing
        self.sensor_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_size = cnn_output_size + 64
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Linear(256, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(256, 1)
        
        # Initialize weights for stable start
        self._init_weights()
        
    def _init_weights(self):
        # Initialize actor mean to output near-zero actions
        if hasattr(self.actor_mean, 'weight'):
            torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            torch.nn.init.constant_(self.actor_mean.bias, 0)
        
        # Initialize actor log_std to reasonable values
        if hasattr(self.actor_log_std, 'weight'):
            torch.nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
            torch.nn.init.constant_(self.actor_log_std.bias, -0.5)
        
        # Initialize critic
        if hasattr(self.critic, 'weight'):
            torch.nn.init.orthogonal_(self.critic.weight, gain=1.0)
            torch.nn.init.constant_(self.critic.bias, 0)
        
    def forward(self, image, sensors):
        # Process image
        x = self.cnn(image)
        x = x.view(x.size(0), -1)  
        
        # Process sensors
        s = self.sensor_fc(sensors)
        
        # Combine features
        combined = torch.cat([x, s], dim=1)
        features = self.fc(combined)
        
        # Actor head - action distribution
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        
        # Clamp log_std to prevent extreme values
        action_log_std = torch.clamp(action_log_std, -5, 2)
        action_std = torch.exp(action_log_std)
        
        # Clamp mean to reasonable range
        action_mean_clamped = torch.tanh(action_mean)  
        
        # Then scale to desired ranges
        steering = action_mean_clamped[:, 0:1] * 0.8 
        throttle = (action_mean_clamped[:, 1:2] + 1.0) * 0.4 + 0.2  
        
        action_mean = torch.cat([steering, throttle], dim=1)
        
        # Critic head - value estimate
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, image, sensors):
        action_mean, action_std, value = self.forward(image, sensors)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        
        # Clamp action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value