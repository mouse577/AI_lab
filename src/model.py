# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCustomCNN(nn.Module):
    def __init__(self):
        super(DeepCustomCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (3, 224, 224) → (32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(2),                # → (32, 112, 112)

            nn.Conv2d(32, 64, 3, padding=1), # → (64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # → (64, 56, 56)

            nn.Conv2d(64, 128, 3, padding=1), # → (128, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(2),                  # → (128, 28, 28)

            nn.Conv2d(128, 256, 3, padding=1), # → (256, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (256, 14, 14)

            nn.Conv2d(256, 512, 3, padding=1), # → (512, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → (512, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(4608, 256),  # ✅ FIXED from 25088 to 4608
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        x = self.conv_block(x)
        print("🧠 Flatten input shape:", x.shape)  # Add this
        x = self.classifier(x)
        return x







