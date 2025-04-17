import torch
import torch.nn as nn

# Defining convulational block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.convblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2), # slope of 0.2
        )
    
    def forward(self, x):
        return self.convblock(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64,128,256,512]):
        super().__init__()
        # BatchNorm not used in the first C64 layer
        self.initialBlock = nn.Sequential(
            # in_channel*2 as it gets both the x and y(y can be real or fake) images concatenated along the channels
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"), 
            nn.LeakyReLU(0.2),
        )
        
        convLayers = []
        in_channels = features[0]
        for feature in features[1:]:
            # stride for last layer is 1
            convLayers.append(CNNBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2))
            in_channels = feature
            
        convLayers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"),)
        
        self.model = nn.Sequential(*convLayers)
        
    def forward(self,x,y):
        x = torch.cat([x,y], dim=1)
        x = self.initialBlock(x)
        x = self.model(x)
        return x

def test():
    x = torch.randn((1, 3, 286, 286))
    y = torch.randn((1, 3, 286, 286))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()