import torch
import torch.nn as nn

"""
: Models for pooling hidden states 
: 1) TimePool (pools over multiple timesteps)
: 2) LayerPool (pools over multiple layers)
"""


class TimePool(nn.Module):
    """
    Pooling layer to aggregate hidden states over multiple timesteps
    """
    def __init__(self, method):
        super().__init__()
        self.method = method

    def forward(self, **kwargs):
        if self.method == 'mean':
            return self.mean_pool(**kwargs)
        elif self.method == 'max':
            return self.max_pool(**kwargs)
        elif self.method == 'both':
            return self.mean_max_pool(**kwargs)
        else:
            raise ValueError("time pooling should be one of mean, max, both.")

    def mean_pool(self, hiddens, input_mask):
        """ Performs mean pooling over timesteps

        Args:
            hiddens: hidden states of last layer                [B x L x E]
            input_mask: masks to avoid [PAD] in computation     [B x L]

        Returns:
            x: sentence representation                          [B x E]

        """
        is_pad = (input_mask == 0)                          # [B x L]
        x = hiddens.float().masked_fill(
            mask=is_pad.unsqueeze(-1),
            value=0.0
        ).type_as(hiddens)                                  # [B x L x E]
        x_lengths = input_mask.sum(dim=-1, keepdims=True)   # [B x 1]
        x = torch.sum(x, dim=1)                             # [B x E]
        x = torch.div(x, x_lengths).type_as(hiddens)        # [B x E]
        return x

    def max_pool(self, hiddens, input_mask):
        """ Performs max pooling over timesteps

        Args:
            hiddens: hidden states of last layer                [B x L x E]
            input_mask: masks to avoid [PAD] in computation     [B x L]

        Returns:
            x: sentence representation                          [B x E]

        """
        is_pad = (input_mask == 0)                          # [B x L]
        x = hiddens.float().masked_fill(
            mask=is_pad.unsqueeze(-1),
            value=float('-inf')
        ).type_as(hiddens)                                  # [B x L x E]
        x_lengths = input_mask.sum(dim=-1, keepdims=True)   # [B x 1]
        x = torch.max(x, dim=1)                             # [B x E]
        x = torch.div(x, x_lengths).type_as(hiddens)        # [B x E]
        return x

    def mean_max_pool(self, **kwargs):
        x_mean = self.mean_pool(**kwargs)           # [B x E]
        x_max = self.max_pool(**kwargs)             # [B x E]
        return torch.cat([x_mean, x_max], dim=1)    # [B x 2E]


class LayerPool(nn.Module):
    def __init__(self, method):
        super().__init__()
        pool_dict = {'avg': self.avg_pool,
                     'max': self.max_pool,
                     'cat': self.concat}
        self.pooler = pool_dict[method]

    def forward(self, hiddens):
        return self.pooler(hiddens)

    def avg_pool(self, hiddens):
        avg_h = torch.stack(hiddens).mean(dim=0)
        return avg_h

    def max_pool(self, hiddens):
        max_h = torch.stack(hiddens).max(dim=0).values
        return max_h

    def concat(self, hiddens):
        return torch.cat(hiddens, dim=1)