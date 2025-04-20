import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CNN(nn.Module):
    def __init__(self, input_dim, channels=16):
        super().__init__()
        # reshape 1×input_dim for conv1d
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(channels*input_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1)           # (B,1,D)
        x = self.conv(x)             # (B,C,D)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, heads=2, layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=heads, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: (B, D) → (D, B, d_model)
        h = self.embed(x).unsqueeze(0)
        h = self.encoder(h)
        return self.fc(h.mean(0)).squeeze(-1)

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.classifier = nn.Linear(bottleneck, 1)
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        y = self.classifier(z).squeeze(-1)
        return y, recon
