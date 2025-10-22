import torch
import torch.nn as nn
import torchvision

from module import DoubleConv, Down, Up

# UNet 모듈도 다시 짜보기
class ChleeUNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, hidden_channel=64):
        super().__init__()
        self.double_conv = DoubleConv(in_channel=in_channel, out_channel=hidden_channel)
        self.down1 = Up(in_channel, hidden_channel)
        self.down2 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(hidden_channel*2, hidden_channel*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(hidden_channel*4, hidden_channel*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(hidden_channel*8, hidden_channel*16))

        nn.Sequential(UpConv(hidden_channel*16, hidden_channel*8), self.crop_and_concat, DoubleConv())



# class ChleeUNet(nn.Module):
#     def __init__(self, in_channel=3, out_channel=1, channel=64):
#         super(ChleeUNet, self).__init__()
        
#         # ============= 1st Conv Block Start =============
#         # input feature shape: (3, 572, 572)
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (64, 570, 570)
#         self.bn1 = nn.BatchNorm2d(num_features=channel)
#         self.relu1 = nn.ReLU(inplace=True)

#         # input feature shape: (64, 570, 570)
#         self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (64, 568, 568)
#         self.bn2 = nn.BatchNorm2d(num_features=channel)
#         self.relu2 = nn.ReLU(inplace=True)

#         # input feature shape: (64, 568, 568)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # output feature shape: (64, 284, 284)

#         # ============= 2nd Conv Block Start =============
#         # input feature shape: (64, 284, 284)
#         self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel*2, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (128, 282, 282)
#         self.bn3 = nn.BatchNorm2d(num_features=channel*2)
#         self.relu3 = nn.ReLU(inplace=True)

#         # input feature shape: (128, 282, 282)
#         self.conv4 = nn.Conv2d(in_channels=channel*2, out_channels=channel*2, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (128, 280, 280)
#         self.bn4 = nn.BatchNorm2d(num_features=channel*2)
#         self.relu4 = nn.ReLU(inplace=True)

#         # input feature shape: (128, 280, 280)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # output feature shape: (128, 140, 140)

#         # ============= 3rd Conv Block Start =============
#         # input feature shape: (128, 140, 140)
#         self.conv5 = nn.Conv2d(in_channels=channel*2, out_channels=channel*4, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (256, 138, 138)
#         self.bn5 = nn.BatchNorm2d(num_features=channel*4)
#         self.relu5 = nn.ReLU(inplace=True)

#         # input feature shape: (256, 138, 138)
#         self.conv6 = nn.Conv2d(in_channels=channel*4, out_channels=channel*4, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (256, 136, 136)
#         self.bn6 = nn.BatchNorm2d(num_features=channel*4)
#         self.relu6 = nn.ReLU(inplace=True)

#         # input feature shape: (256, 136, 136)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # output feature shape: (256, 68, 68)

#         # ============= 4rd Conv Block Start =============
#         # input feature shape: (256, 68, 68)
#         self.conv7 = nn.Conv2d(in_channels=channel*4, out_channels=channel*8, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (512, 66, 66)
#         self.bn7 = nn.BatchNorm2d(num_features=channel*8)
#         self.relu7 = nn.ReLU(inplace=True)

#         # input feature shape: (512, 66, 66)
#         self.conv8 = nn.Conv2d(in_channels=channel*8, out_channels=channel*8, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (512, 64, 64)
#         self.bn8 = nn.BatchNorm2d(num_features=channel*8)
#         self.relu8 = nn.ReLU(inplace=True)

#         # input feature shape: (512, 64, 64)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # output feature shape: (512, 32, 32)

#         # ============= 5rd Conv Block Start =============
#         # input feature shape: (512, 32, 32)
#         self.conv9 = nn.Conv2d(in_channels=channel*8, out_channels=channel*16, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (1024, 30, 30)
#         self.bn9 = nn.BatchNorm2d(num_features=channel*16)
#         self.relu9 = nn.ReLU(inplace=True)

#         # input feature shape: (1024, 30, 30)
#         self.conv10 = nn.Conv2d(in_channels=channel*16, out_channels=channel*16, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (1024, 28, 28)
#         self.bn10 = nn.BatchNorm2d(num_features=channel*16)
#         self.relu10 = nn.ReLU(inplace=True)


#         # ============= 1st Up-Conv Block Start =============
#         # input feature shape: (1024, 28, 28)
#         self.upconv1 = nn.ConvTranspose2d(in_channels=channel*16, out_channels=channel*8, kernel_size=2, stride=2, padding=0)
#         # "copy and crop" concat → output feature shape: (1024, 56, 56)
#         self.conv11 = nn.Conv2d(in_channel=(channel*8)*2, out_channels=channel*8, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (512, 54, 54)
#         self.bn11 = nn.BatchNorm2d(num_features=channel*8)
#         self.relu11 = nn.ReLU(inplace=True)

#         # input feature shape: (512, 54, 54)
#         self.conv12 = nn.Conv2d(in_channels=channel*8, out_channels=channel*8, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (512, 52, 52)
#         self.bn12 = nn.BatchNorm2d(num_features=channel*8)
#         self.relu12 = nn.ReLU(inplace=True)

#         # ============= 2nd Up-Conv Block Start =============
#         # input feature shape: (512, 52, 52)
#         self.upconv2 = nn.ConvTranspose2d(in_channels=channel*8, out_channels=channel*4, kernel_size=2, stride=2, padding=0)
#         # "copy and crop" concat → output feature shape: (512, 104, 104)
#         self.conv13 = nn.Conv2d(in_channels=(channel*4)*2, out_channels=channel*4, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (256, 102, 102)
#         self.bn13 = nn.BatchNorm2d(num_features=channel*4)
#         self.relu13 = nn.ReLU(inplace=True)

#         # input feature shape: (256, 102, 102)
#         self.conv14 = nn.Conv2d(in_channels=channel*4, out_channels=channel*4, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (256, 100, 100)
#         self.bn14 = nn.BatchNorm2d(num_features=channel*4)
#         self.relu14 = nn.ReLU(inplace=True)

#         # ============= 3rd Up-Conv Block Start =============
#         # input feature shape: (256, 100, 100)
#         self.upconv3 = nn.ConvTranspose2d(in_channels=channel*4, out_channels=channel*2, kernel_size=2, stride=2, padding=0)
#         # "copy and crop" concat → output feature shape: (256, 200, 200)
#         self.conv15 = nn.Conv2d(in_channels=(channel*2)*2, out_channels=channel*2, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (128, 198, 198)
#         self.bn15 = nn.BatchNorm2d(num_features=channel*2)
#         self.relu15 = nn.ReLU(inplace=True)

#         # input feature shape: (128, 198, 198)
#         self.conv16 = nn.Conv2d(in_channels=channel*2, out_channels=channel*2, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (128, 196, 196)
#         self.bn16 = nn.BatchNorm2d(num_features=channel*2)
#         self.relu16 = nn.ReLU(inplace=True)

#         # ============= 4rd Up-Conv Block Start =============
#         # input feature shape: (128, 196, 196)
#         self.upconv4 = nn.ConvTranspose2d(in_channels=channel*2, out_channels=channel, kernel_size=2, stride=2, padding=0)
#         # "copy and crop" concat → output feature shape: (128, 392, 392)
#         self.conv17 = nn.Conv2d(in_channels=(channel)*2, out_channels=channel, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (64, 390, 390)
#         self.bn17 = nn.BatchNorm2d(num_features=channel)
#         self.relu17 = nn.ReLU(inplace=True)

#         # input feature shape: (64, 390, 390)
#         self.conv18 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=0)
#         # output feature shape: (64, 388, 388)
#         self.bn18 = nn.BatchNorm2d(num_features=channel)
#         self.relu18 = nn.ReLU(inplace=True)

#         # input feature shape: (64, 388, 388)
#         self.conv19 = nn.Conv2d(in_channels=channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
#         # output feature shape: (1, 388, 388)

#     def crop_and_concat(self, encoder_feature, decoder_feature):
#         '''
#         conv shape: [B, C, H, W]
#         '''
#         _, _, H, W = decoder_feature.size()
#         cropped_encoder_feature = torchvision.transforms.CenterCrop([H, W])(encoder_feature)
#         return torch.cat([cropped_encoder_feature, decoder_feature], dim=1)


#     def forward(self, x):
#         # Encoder
#         conv1 = self.relu1(self.bn1(self.conv1(x)))
#         conv2 = self.relu2(self.bn2(self.conv2(conv1)))
#         pool1 = self.pool1(conv2)

#         conv3 = self.relu3(self.bn3(self.conv3(pool1)))
#         conv4 = self.relu4(self.bn4(self.conv4(conv3)))
#         pool2 = self.pool2(conv4)

#         conv5 = self.relu5(self.bn5(self.conv5(pool2)))
#         conv6 = self.relu6(self.bn6(self.conv6(conv5)))
#         pool3 = self.pool3(conv6)

#         conv7 = self.relu7(self.bn7(self.conv7(pool3)))
#         conv8 = self.relu8(self.bn8(self.conv8(conv7)))
#         pool4 = self.pool4(conv8)

#         conv9 = self.relu9(self.bn9(self.conv9(pool4)))
#         conv10 = self.relu10(self.bn10(self.conv10(conv9)))

#         # Decoder 
#         upconv1 = self.upconv1(conv10)
#         concat_upconv1 = self.crop_and_concat(conv8, upconv1)
#         conv11 = self.relu11(self.bn11(self.conv11(concat_upconv1)))
#         conv12 = self.relu12(self.bn12(self.conv12(conv11)))

#         upconv2 = self.upconv2(conv12)
#         concat_upconv2 = self.crop_and_concat(conv6, upconv2)
#         conv13 = self.relu13(self.bn13(self.conv13(concat_upconv2)))
#         conv14 = self.relu14(self.bn14(self.conv14(conv13)))

#         upconv3 = self.upconv3(conv14)
#         concat_upconv3 = self.crop_and_concat(conv4, upconv3)
#         conv15 = self.relu15(self.bn15(self.conv15(concat_upconv3)))
#         conv16 = self.relu16(self.bn16(self.conv16(conv15)))

#         upconv4 = self.upconv4(conv16)
#         concat_upconv4 = self.crop_and_concat(conv2, upconv4)
#         conv17 = self.relu17(self.bn17(self.conv17(concat_upconv4)))
#         conv18 = self.relu18(self.bn18(self.conv18(conv17)))

#         return self.conv19(conv18)