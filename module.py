import torch
import torch.nn as nn
import torchvision


class DoubleConv(nn.Module):
    """
    Double Conv, BatchNorm, ReLu (CBR)
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.down(x)
    

# Up 모듈 다시 짜보기
class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, 2)
        self.double_conv = DoubleConv(in_channel, out_channel)

    def forward(self, x, skip):
        x = self.up(x)

        if x.size()[2:] != skip.size()[2:]:
            skip = torchvision.transforms.functional.center_crop(skip, x.size()[2:])

        x = torch.cat([skip, x], dim=1)

        return self.double_conv(x)