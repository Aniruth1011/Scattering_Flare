import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LensBlock(nn.Module):
    def __init__(self , input_dim):
        super(LensBlock, self).__init__()

        self.conv1 = nn.Conv3d(256, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock3D(64, 64)
        self.residual_block2 = ResidualBlock3D(64, 128)
        self.conv2 = nn.Conv3d(128, input_dim, kernel_size=3, padding=1)
        
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.conv1(x)

        print("con1" , x.shape)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.conv2(x)
        
        x = self.tanh(x)

        return x