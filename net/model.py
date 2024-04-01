import torch
import torch.nn as nn 
from net.block import DoubleConv , Down , TransformerBlock , Up, LensComponent

class UNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3):
        super(UNet, self).__init__()
        self.inc = DoubleConv(input_channels, 64)
        self.enc_lev_1 = Down(64, 128)
        self.transformer1 = TransformerBlock(128)
        self.enc_lev_2 = Down(128, 256)
        self.transformer2 = TransformerBlock(256)
        self.enc_lev_3 = Down(256, 512)
        self.transformer3 = TransformerBlock(512)

        self.lens3 = LensComponent()
        self.dec_lev_3 = Up(768, 256)
        self.transformer4 = TransformerBlock(256)

        self.lens2 = LensComponent()
        self.dec_lev_2 = Up(384, 128)
        self.transformer5 = TransformerBlock(128)

        self.lens1 = LensComponent()
        self.dec_lev_1 = Up(192, 64)
        self.transformer6 = TransformerBlock(64)

        self.outc = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):

        x_enc_lev_0 = self.inc(x)

        x_enc_lev_1 = self.enc_lev_1(x_enc_lev_0)
        x_enc_lev_1 = self.transformer1(x_enc_lev_1)
        # print(x_enc_lev_1.shape)
        # print('STEP 1')

        x_enc_lev_2 = self.enc_lev_2(x_enc_lev_1)
        x_enc_lev_2 = self.transformer2(x_enc_lev_2)
        # print(x_enc_lev_2.shape)
        # print('STEP 2')

        x_enc_lev_3 = self.enc_lev_3(x_enc_lev_2)
        x_enc_lev_3 = self.transformer3(x_enc_lev_3)
        # print(x_enc_lev_3.shape)
        # print('STEP 3')

        x_dec_lens_lev_3 = self.lens3(x_enc_lev_3)
        # pt_1 = x_dec_lens_lev_3.view(-1)
        # pt_2 = x_enc_lev_3.view(-1)
        device1 = x_dec_lens_lev_3.device
        device2 = x_enc_lev_3.device
        # Move tensor1 to the same device as tensor2
        if device1 != device2:
            x_dec_lens_lev_3 = x_dec_lens_lev_3.to(device2)
        if x_dec_lens_lev_3.view(-1).size(0) < x_enc_lev_3.view(-1).size(0):
            pt_1 = torch.cat([x_dec_lens_lev_3.view(-1), torch.zeros(x_enc_lev_3.view(-1).size(0) - x_dec_lens_lev_3.view(-1).size(0), device=x_enc_lev_3.device)])
            pt_2 = x_enc_lev_3.view(-1)
        else:
            pt_1 = x_dec_lens_lev_3.view(-1)
            pt_2 = torch.cat([x_enc_lev_3.view(-1), torch.zeros(x_dec_lens_lev_3.view(-1).size(0) - x_enc_lev_3.view(-1).size(0), device=x_enc_lev_3.device)])

        # x_dec_lev_3 = torch.dot(x_dec_lens_lev_3.view(-1), x_enc_lev_3.view(-1))
        x_dec_lev_3 = torch.dot(pt_1, pt_2)
        x_dec_lev_3 = self.dec_lev_3(x_dec_lev_3, x_enc_lev_2)
        x_dec_lev_3 = self.transformer4(x_dec_lev_3)
        # x_dec_lens_lev_3 = x_lens_3 * x_enc_lev_3
        # print(x_enc_lev_3.shape)
        # print('STEP 4')

        x_dec_lens_lev_2 = self.lens2(x_dec_lev_3)
        x_dec_lev_2 = torch.dot(x_dec_lens_lev_2, x_dec_lev_3)
        x_dec_lev_2 = self.dec_lev_2(x_dec_lev_2, x_enc_lev_1)
        x_dec_lev_2 = self.transformer5(x_dec_lev_2)
        # print(x_enc_lev_1.shape)
        # print('STEP 5')

        x_dec_lens_lev_1 = self.lens1(x_enc_lev_2)
        x_dec_lev_1 = torch.dot(x_dec_lens_lev_1, x_enc_lev_2)
        x_dec_lev_1 = self.dec_lev_1(x_dec_lev_1, x_enc_lev_0)
        x_dec_lev_1 = self.transformer6(x_dec_lev_1)
        # print(x_enc_lev_1.shape)
        # print('STEP 6')

        x_out = self.outc(x_dec_lev_1)
        # print('SUCCESS\n')
        return x_out
