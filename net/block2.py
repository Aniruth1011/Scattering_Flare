import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        out = torch.mul(channel_out, spatial_out)
        return out

class CombinedAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CombinedAttention, self).__init__()
        self.attention = ChannelSpatialAttention(in_channels, ratio, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.conv_final = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)  # Updated to include input channels

    def forward(self, x):
        att = self.attention(x)
        conv_out = self.conv(att)
        tanh_out = self.tanh(conv_out)
        combined = torch.cat([tanh_out, x], dim=1)  # Concatenate along channel dimension
        final_out = self.conv_final(combined)  # Apply conv to match input size
        return final_out
