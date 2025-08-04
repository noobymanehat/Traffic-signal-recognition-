import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignNet(nn.Module):
    """
    Input: 32x32x3
    """
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout Layers
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)
        
        # Fully Connected Layers
        # After 3 pooling layers (32/2/2/2 = 4), the image size is 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # No need for softmax here if using CrossEntropyLoss
        # CrossEntropyLoss applies softmax internally
        return x