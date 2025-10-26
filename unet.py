import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. 기본 블록 (벽돌)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    (3x3 Conv + BatchNorm + ReLU) * 2
    논문 스타일의 padding=0 (Valid Convolution) 사용
    """
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
        self.conv = DoubleConv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)  # <-- 스킵 연결용
        pooled = self.pool(conv_out)
        return conv_out, pooled


class Up(nn.Module):
    """
    ConvTranspose2d -> Crop & Concat -> DoubleConv
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 업샘플링: 채널은 절반(in_channel -> out_channel), 크기는 2배
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        
        # Concat 이후: (스킵 채널 + 업샘플링 채널) = out_channel * 2
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

        # --- 수축 경로 (Encoder) ---
        # (B, 3, H, W)
        self.encoder1 = Encoder(in_channel, hidden_channel) # 3 -> 64
        # (B, 64, H, W) -> s1 (64c), p1 (64c)
        
        self.encoder2 = Encoder(hidden_channel, hidden_channel*2) # 64 -> 128
        # (B, 128, H, W) -> s2 (128c), p2 (128c)
        
        self.encoder3 = Encoder(hidden_channel*2, hidden_channel*4) # 128 -> 256
        # (B, 256, H, W) -> s3 (256c), p3 (256c)
        
        self.encoder4 = Encoder(hidden_channel*4, hidden_channel*8) # 256 -> 512
        # (B, 512, H, W) -> s4 (512c), p4 (512c)

        # --- 병목 구간 (Bottleneck) ---
        self.bottleneck = DoubleConv(hidden_channel*8, hidden_channel*16) # 512 -> 1024
        
        # --- 확장 경로 (Decoder) ---
        self.up1 = Up(hidden_channel*16, hidden_channel*8) # 1024 -> 512
        self.up2 = Up(hidden_channel*8, hidden_channel*4) # 512 -> 256
        self.up3 = Up(hidden_channel*4, hidden_channel*2) # 256 -> 128
        self.up4 = Up(hidden_channel*2, hidden_channel) # 128 -> 64

        # --- 최종 출력 (Output) ---
        self.out_conv = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

    def forward(self, x):
        # --- 수축 ---
        s1, p1 = self.encoder1(x)  # s1: 64c, p1: 64c
        s2, p2 = self.encoder2(p1) # s2: 128c, p2: 128c
        s3, p3 = self.encoder3(p2) # s3: 256c, p3: 256c
        s4, p4 = self.encoder4(p3) # s4: 512c, p4: 512c

        # --- 병목 ---
        b = self.bottleneck(p4)     # b: 1024c

        # --- 확장 & 스킵 연결 ---
        x = self.up1(b, s4)   # (1024c, 512c) -> 512c
        x = self.up2(x, s3)   # (512c, 256c) -> 256c
        x = self.up3(x, s2)   # (256c, 128c) -> 128c
        x = self.up4(x, s1)   # (128c, 64c) -> 64c

        # --- 최종 출력 ---
        output = self.out_conv(x) # 64c -> out_channel (1c)
        
        return output