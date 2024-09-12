import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=512):
        super(PositionalEncoding, self)._init_()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, :, 1::2] = torch.cos(position * div_term)

# Define TransUNet model
class TransUNet(nn.Module):
    def _init_(self, in_channels, out_channels, patch_size, img_size, num_classes, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransUNet, self)._init_()

        # Embedding layer
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=(img_size // patch_size) ** 2)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output layer
        self.output_layer = nn.Conv2d(d_model, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x)
        x = x.permute(1, 2, 0).view(x.size(1), x.size(2), -1)
        x = self.output_layer(x)
        return x

# Instantiate the model
model = TransUNet(in_channels=3, out_channels=1, patch_size=16, img_size=256, num_classes=2, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

# Example usage
input_data = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 input
output = model(input_data)