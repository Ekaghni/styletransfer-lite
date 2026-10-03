import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, num_features, num_styles):
        super().__init__()
        self.num_styles = num_styles
        self.gamma = nn.Parameter(torch.ones(num_styles, num_features))
        self.beta = nn.Parameter(torch.zeros(num_styles, num_features))
    
    def forward(self, x, style_weights):
        # style_weights: [num_styles] summing to 1
        gamma = (style_weights @ self.gamma).view(1, -1, 1, 1)
        beta = (style_weights @ self.beta).view(1, -1, 1, 1)
        
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized = (x - mean) / std
        return gamma * normalized + beta

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_styles):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.norm1 = ConditionalInstanceNorm(channels, num_styles)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.norm2 = ConditionalInstanceNorm(channels, num_styles)
    
    def forward(self, x, style_weights):
        residual = x
        out = F.relu(self.norm1(self.conv1(x), style_weights))
        out = self.norm2(self.conv2(out), style_weights)
        return out + residual

class StyleTransferNet(nn.Module):
    def __init__(self, num_styles=26):
        super().__init__()
        self.num_styles = num_styles
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 9, padding=4, padding_mode='reflect')
        self.norm1 = ConditionalInstanceNorm(32, num_styles)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect')
        self.norm2 = ConditionalInstanceNorm(64, num_styles)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect')
        self.norm3 = ConditionalInstanceNorm(128, num_styles)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(128, num_styles) for _ in range(5)])
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.norm4 = ConditionalInstanceNorm(64, num_styles)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.norm5 = ConditionalInstanceNorm(32, num_styles)
        self.conv_out = nn.Conv2d(32, 3, 9, padding=4, padding_mode='reflect')
    
    def forward(self, x, style_weights):
        # Encoder
        out = F.relu(self.norm1(self.conv1(x), style_weights))
        out = F.relu(self.norm2(self.conv2(out), style_weights))
        out = F.relu(self.norm3(self.conv3(out), style_weights))
        
        # Residual
        for block in self.res_blocks:
            out = block(out, style_weights)
        
        # Decoder
        out = F.relu(self.norm4(self.deconv1(out), style_weights))
        out = F.relu(self.norm5(self.deconv2(out), style_weights))
        out = torch.sigmoid(self.conv_out(out))
        return out

# Export for Android
model = StyleTransferNet(num_styles=26)
model.eval()

# You'll need to load weights from the .pb file or train fresh
example_input = torch.randn(1, 3, 500, 500)
example_style = torch.zeros(26)
example_style[0] = 1.0

traced = torch.jit.trace(model, (example_input, example_style))
traced.save("style_transfer.pt")