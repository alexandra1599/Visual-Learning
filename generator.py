import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout == True:
            return self.dropout(x)
        else:
            return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) #128
        
        self.down1 = ConvBlock(features, features*2, down=True,act="leaky",use_dropout=False) #64
        self.down2 = ConvBlock(features*2, features*4, down=True,act="leaky",use_dropout=False) #32
        self.down3 = ConvBlock(features*4, features*8, down=True,act="leaky",use_dropout=False) #16
        self.down4 = ConvBlock(features*8, features*8, down=True,act="leaky",use_dropout=False) #8
        self.down5 = ConvBlock(features*8, features*8, down=True,act="leaky",use_dropout=False) #4
        self.down6 = ConvBlock(features*8, features*8, down=True,act="leaky",use_dropout=False) #2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.ReLU(),
        )  # 1x1
        
        self.up1 = ConvBlock(features*8, features*8, down=False,act="relu",use_dropout=True)
        self.up2 = ConvBlock(features*16, features*8, down=False,act="relu",use_dropout=True)
        self.up3 = ConvBlock(features*16, features*8, down=False,act="relu",use_dropout=True)
        self.up4 = ConvBlock(features*16, features*8, down=False,act="relu",use_dropout=False)
        self.up5 = ConvBlock(features*16, features*4, down=False,act="relu",use_dropout=False)
        self.up6 = ConvBlock(features*8, features*2, down=False,act="relu",use_dropout=False)
        self.up7 = ConvBlock(features*4, features, down=False,act="relu",use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        d0 = self.initial_down(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleneck = self.bottleneck(d6)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1,d6],1))
        u3 = self.up3(torch.cat([u2,d5],1))
        u4 = self.up4(torch.cat([u3,d4],1))
        u5 = self.up5(torch.cat([u4,d3],1))
        u6 = self.up6(torch.cat([u5,d2],1))
        u7 = self.up7(torch.cat([u6,d1],1))
        final_up = self.final_up(torch.cat([u7,d0],1))

        return final_up

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()