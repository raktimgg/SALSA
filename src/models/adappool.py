import torch
from torch import nn
from torch.nn import functional as F

class AdaptivePooling(nn.Module):
    def __init__(self, feature_dim, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.query = nn.Parameter(torch.randn(output_channels, feature_dim))

    def forward(self, x, return_weights=False):
        """
        Args:
            x: Input tensor of shape (batch_size, input_channels, feature_dim)

        Returns:
            Output tensor of shape (batch_size, output_channels, feature_dim)
        """
        query = self.query.unsqueeze(0).repeat(x.shape[0],1,1)

        out = F.scaled_dot_product_attention(query=query,key=x,value=x)
        if return_weights:
            attn_scores = torch.einsum('ij,bkj->bki', self.query, x)
            attn_weights = F.softmax(attn_scores, dim=1)
            return out, attn_weights

        return out
