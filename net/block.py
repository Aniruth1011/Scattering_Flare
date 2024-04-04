import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LensComponent(nn.Module):
    def __init__(self, input_dim, pretrained=True):
        super(LensComponent, self).__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim)) 
        nn.init.normal_(self.weight) 
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(input_dim , input_dim , kernel_size=3, padding=1 )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Apply convolutional layer
        conv_output_1 = self.conv1(x)

        conv_output_2 = self.conv2(conv_output_1)
        
        # Apply the weight parameter to the convolutional output
        weighted_output = conv_output_2 * self.weight.view(1, self.input_dim, 1, 1)
        
        # Apply activation function
        output = self.tanh(weighted_output)
        
        return output



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
        # print('1. ChennelAttention')
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
        # print('2. SpatialAttention')
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AttentionFusion(nn.Module):
    def __init__(self, in_planes):
        super(AttentionFusion, self).__init__()
        self.fc = nn.Linear(in_planes * 2, in_planes)

    # def forward(self, channel_attention, spatial_attention):
    #     spatial_attention = F.interpolate(spatial_attention, size=channel_attention.size()[2:], mode='nearest')
    #     combined_attention = torch.cat([channel_attention, spatial_attention], dim=1)
    #     combined_attention = self.fc(combined_attention)
    #     return combined_attention
        
    def forward(self, channel_attention, spatial_attention):
        # print('3. AttentionFusion')
        spatial_attention = F.interpolate(spatial_attention, size=channel_attention.size()[2:], mode='nearest')
        if spatial_attention.size() != channel_attention.size():
            spatial_attention = F.pad(spatial_attention, (0, 0, 0, 128 - spatial_attention.size(1)), mode='constant', value=0)
        result = spatial_attention * channel_attention
        return result
    
    # def forward(self, channel_attention, spatial_attention):
    #     spatial_attention = F.interpolate(spatial_attention, size=channel_attention.size()[2:], mode='nearest')
    #     if spatial_attention.size() != channel_attention.size():
    #         spatial_attention = F.pad(spatial_attention, (0, 0, 0, channel_attention.size(1) - spatial_attention.size(1)), mode='constant', value=0)
    #     combined_attention = torch.cat([channel_attention, spatial_attention], dim=1)
    #     combined_attention = self.fc(combined_attention)
    #     return combined_attention

class TransformerBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(TransformerBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.attention_fusion = AttentionFusion(in_planes)

    # def forward(self, x):
    #     channel_attention = self.channel_attention(x)
    #     spatial_attention = self.spatial_attention(x)
    #     combined_attention = self.attention_fusion(channel_attention, spatial_attention)
    #     x = x * combined_attention
    #     return x
        
    def forward(self, x):
        # print('4. TransformerBlock')
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        combined_attention = self.attention_fusion(channel_attention, spatial_attention)
        combined_attention = F.interpolate(combined_attention, size=x.size()[2:], mode='nearest')
        x = x * combined_attention
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print('5. DoubleConv')
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # print('6. Down')
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print('7. UP')
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)