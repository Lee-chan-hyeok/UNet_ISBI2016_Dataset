import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. 기본 블록 (벽돌)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    DoubleConv -> MaxPool
    2개의 값을 반환:
    1. conv_out: 스킵 연결(Skip Connection)을 위한 피처맵
    2. pooled: 다음 Encoder 블록으로 전달될 다운샘플링된 피처맵
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = DoubleConv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        double_conv_out = self.double_conv(x)  # skip connection꺼
        pool_out = self.pool(double_conv_out)
        return double_conv_out, pool_out


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channel * 2, out_channel)

    def crop_center(self, enc_feat, dec_feat):
        """
        ex)
            enc_feat: (batch_size, 512, 64, 64)
            dec_feat: (batch_size, 1024, 56, 56)
        """
        _, _, H_dec, W_dec = dec_feat.size()
        _, _, H_enc, W_enc = enc_feat.size()

        delta_H = (H_enc - H_dec) // 2
        delta_W = (W_enc - W_dec) // 2
        
        # [:, :, H시작:H끝, W시작:W끝]
        return enc_feat[:, :, delta_H:delta_H + H_dec, delta_W:delta_W + W_dec]

    def forward(self, x, enc_feat):
        # x: 아래(bottleneck)에서 올라온 피처맵
        # enc_feat: 수축 경로(Encoder)에서 온 스킵 연결 피처맵

        # 1. 업샘플링
        x = self.up_conv(x)
        
        # 2. Crop & Concat
        # 스킵 연결(enc_feat)이 업샘플링된(x) 것보다 크므로, 중앙을 잘라냄
        if enc_feat.shape[-2:] != x.shape[-2:]:
            enc_cropped = self.crop_center(enc_feat, x)
        else:
            enc_cropped = enc_feat
            
        # 채널(dim=1) 기준으로 합치기
        x = torch.cat([enc_cropped, x], dim=1)
        
        # 3. DoubleConv
        x = self.double_conv(x)
        return x


class ChleeUNet(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=64, out_channel=1):
        super().__init__()

        # Contracting path
        self.encoder1 = Encoder(in_channel, hidden_channel) # 3 -> 64
        self.encoder2 = Encoder(hidden_channel, hidden_channel*2) # 64 -> 128
        self.encoder3 = Encoder(hidden_channel*2, hidden_channel*4) # 128 -> 256
        self.encoder4 = Encoder(hidden_channel*4, hidden_channel*8) # 256 -> 512

        self.double_conv = DoubleConv(hidden_channel*8, hidden_channel*16) # 512 -> 1024
        
        # Expanding path
        self.decoder1 = Decoder(hidden_channel*16, hidden_channel*8) # 1024 -> 512
        self.decoder2 = Decoder(hidden_channel*8, hidden_channel*4) # 512 -> 256
        self.decoder3 = Decoder(hidden_channel*4, hidden_channel*2) # 256 -> 128
        self.decoder4 = Decoder(hidden_channel*2, hidden_channel) # 128 -> 64

        # 1x1 conv
        self.out_conv = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

    def forward(self, x):
        double_conv_out1, pool_out1 = self.encoder1(x)          # double_conv_out1, pool_out1: 64C
        double_conv_out2, pool_out2 = self.encoder2(pool_out1)  # double_conv_out2, pool_out2: 128C
        double_conv_out3, pool_out3 = self.encoder3(pool_out2)  # double_conv_out3, pool_out3: 256C
        double_conv_out4, pool_out4 = self.encoder4(pool_out3)  # double_conv_out4, pool_out4: 512C

        double_conv_out5 = self.double_conv(pool_out4)

        dec_out = self.decoder1(double_conv_out5, double_conv_out4)
        dec_out = self.decoder2(dec_out, double_conv_out3)
        dec_out = self.decoder3(dec_out, double_conv_out2)
        dec_out = self.decoder4(dec_out, double_conv_out1)

        output = self.out_conv(dec_out)
        
        return output