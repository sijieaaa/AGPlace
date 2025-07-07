

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: [b, c, h, w]
        output = nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        output = output.view(x.size(0), -1)
        return output




class ConvAP(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels=512, s1=2, s2=2):
        super(ConvAP, self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x):
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x
    

class CosPlace(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor 
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        # x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2) # [b,c,hw]
        x = self.mix(x) # [b,c,hw]
        x = x.permute(0, 2, 1) # [b,hw,c]
        x = self.channel_proj(x) # [b,hw,c]
        x = x.permute(0, 2, 1) # [b,c,hw]
        x = self.row_proj(x) # [b,c,out_rows]
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper 
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale 
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """
    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.ln2(out)
        return out



if __name__ == '__main__':

    b = 4
    c = 256
    h = 10
    w = 10

    x = torch.randn(b, c, h, w)
    gem = GeM()
    mixvpr = MixVPR(
        in_channels=c,
        in_h=h,
        in_w=w,
        out_channels=c,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4
    )
    convap = ConvAP(
        in_channels=c,
        out_channels=c,
        s1=2,
        s2=2 
    )
    cosplace = CosPlace(
        in_dim=c,
        out_dim=c
    )
    rrm = RRM(dim=c)


    output_gem = gem(x)
    output_mixvpr = mixvpr(x)
    output_convap = convap(x)
    output_cosplace = cosplace(x)
    output_rrm = rrm(x)


    print("output_gem.shape:", output_gem.shape)
    print("output_mixvpr.shape:", output_mixvpr.shape)
    print("output_convap.shape:", output_convap.shape)
    print("output_cosplace.shape:", output_cosplace.shape)
    print("output_rrm.shape:", output_rrm.shape)

