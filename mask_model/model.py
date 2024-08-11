import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskComputationModel(nn.Module):
    def __init__(self, text_embedding_dim, noise_channels, time_embedding_dim, hidden_dim=128):
        super(MaskComputationModel, self).__init__()
        
        # CNN for noise image
        self.conv1 = nn.Conv2d(noise_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # MLP for text guidance
        self.text_mlp = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion and output
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)  # Output single-channel mask

    def forward(self, noise_img, text_embedding, time_step):
        # Process noise image through CNN
        x = F.relu(self.conv1(noise_img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Process text embedding through MLP
        text_features = self.text_mlp(text_embedding)
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        text_features = text_features.expand(-1, -1, x.size(2), x.size(3))  # Expand to match image feature size
        
        # Process time step embedding
        time_embedding = self.time_embed(time_step.unsqueeze(-1))
        time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-1)
        time_embedding = time_embedding.expand(-1, -1, x.size(2), x.size(3))
        
        # Fuse features
        x = x + text_features + time_embedding
        
        # Further processing and generate mask
        x = F.relu(self.conv4(x))
        mask = torch.sigmoid(self.conv5(x))  # Output mask with values in [0, 1]
        
        return mask