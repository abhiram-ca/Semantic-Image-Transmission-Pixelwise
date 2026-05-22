import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

class SPADEGenerator(nn.Module):
    def __init__(self, semantic_nc, z_dim=256):
        super().__init__()
        self.semantic_nc = semantic_nc
        self.z_dim = z_dim
        
        # Initial resolution 6x6
        self.fc = nn.Linear(z_dim, 1024 * 6 * 6) 
        
        self.head_0 = SPADEResnetBlock(1024, 1024, semantic_nc)
        
        self.G_middle_0 = SPADEResnetBlock(1024, 1024, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024, semantic_nc)
        
        self.up_0 = SPADEResnetBlock(1024, 512, semantic_nc)
        self.up_1 = SPADEResnetBlock(512, 256, semantic_nc)
        self.up_2 = SPADEResnetBlock(256, 128, semantic_nc)
        self.up_3 = SPADEResnetBlock(128, 64, semantic_nc)
        
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, seg, z=None):
        if z is None:
            z = torch.randn(seg.size(0), self.z_dim, dtype=seg.dtype, device=seg.device)
            
        x = self.fc(z)
        x = x.view(-1, 1024, 6, 6)
        
        x = self.head_0(x, seg)
        
        x = self.up(x) # 12x12
        x = self.up_0(x, seg)
        
        x = self.up(x) # 24x24
        x = self.up_1(x, seg)
        
        x = self.up(x) # 48x48
        x = self.up_2(x, seg)
        
        x = self.up(x) # 96x96
        x = self.up_3(x, seg)
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult), 
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128x128
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 64x64
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(True),
            nn.Conv2d(512, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Normalize features
        )
        
    def forward(self, x):
        return self.model(x)

class SPADEGenerator256(nn.Module):
    def __init__(self, semantic_nc, z_dim=256):
        super().__init__()
        self.semantic_nc = semantic_nc
        self.z_dim = z_dim
        
        # Initial resolution 8x8
        self.fc = nn.Linear(z_dim, 1024 * 8 * 8) 
        
        self.head_0 = SPADEResnetBlock(1024, 1024, semantic_nc)
        
        self.up_0 = SPADEResnetBlock(1024, 1024, semantic_nc)
        self.up_1 = SPADEResnetBlock(1024, 512, semantic_nc)
        self.up_2 = SPADEResnetBlock(512, 256, semantic_nc)
        self.up_3 = SPADEResnetBlock(256, 128, semantic_nc)
        self.up_4 = SPADEResnetBlock(128, 64, semantic_nc)
        
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, seg, z=None):
        if z is None:
            z = torch.randn(seg.size(0), self.z_dim, dtype=seg.dtype, device=seg.device)
            
        x = self.fc(z)
        x = x.view(-1, 1024, 8, 8)
        
        x = self.head_0(x, seg)
        
        x = self.up(x) # 16x16
        x = self.up_0(x, seg)
        
        x = self.up(x) # 32x32
        x = self.up_1(x, seg)
        
        x = self.up(x) # 64x64
        x = self.up_2(x, seg)
        
        x = self.up(x) # 128x128
        x = self.up_3(x, seg)
        
        x = self.up(x) # 256x256
        x = self.up_4(x, seg)
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

