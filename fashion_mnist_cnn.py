import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        
        # First convolutional layer followed by ReLU and MaxPooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second convolutional layer followed by ReLU and MaxPooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        # Passing through the first conv-relu-pool block
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Passing through the second conv-relu-pool block
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the tensor before passing it to fully connected layers
        x = x.view(-1, 64*7*7)
        
        # Passing through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = FashionMNISTCNN()
    print(model)
