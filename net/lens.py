# import torch
# import torch.nn as nn

# class LensComponent(nn.Module):
#     def __init__(self, resnet, transformer):
#         super(LensComponent, self).__init__()
#         self.resnet = resnet
#         self.transformer = transformer
#         self.flatten = nn.Flatten()
#         self.tanh = nn.Tanh()

#     def forward(self, x_resnet, x_transformer):
#         features_resnet = self.resnet(x_resnet)
#         flattened_features = self.flatten(features_resnet)
#         tanh_output = self.tanh(flattened_features)
#         # lens_output = torch.dot(tanh_output, x_transformer)
        
#         return tanh_output

# resnet_model = ...  # model weights
# transformer_model = 'ckpt/unet_model.pth'  # trans weights

# # Initialize LensComponent
# lens_component = LensComponent(resnet=resnet_model, transformer=transformer_model)

# resnet_input = torch.randn(1, 512, 128, 128)
# # resnet_input = (1, 128, 128, 128)
# transformer_input = torch.randn(1, trans_size_podanum) 
# output = lens_component(resnet_input, transformer_input)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LensComponent(nn.Module):
    def __init__(self):
        super(LensComponent, self).__init__()
        # Load pre-trained ResNet
        self.resnet = models.resnet18(pretrained=True)
        # Disable gradient computation for ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()

    def forward(self, x_transformer):
        # Pass transformer output through ResNet
        features_resnet = self.resnet(x_transformer)
        flattened_features = self.flatten(features_resnet)
        tanh_output = self.tanh(flattened_features)
        return tanh_output

# Example usage
# Assuming x_transformer is your input tensor from previous transformer layers
x_transformer = torch.randn(1, 3, 224, 224)  # Example input tensor
lens_component = LensComponent()
output = lens_component(x_transformer)
print(output.shape)  # Print the shape of the output tensor
