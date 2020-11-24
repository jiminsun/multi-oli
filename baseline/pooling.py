import torch
import torch.nn as nn

"""
: Models for pooling over multiple timesteps of hidden states 
: 1) MeanPool
: 2) MaxPool
: 3) MeanMaxPool
"""


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        """ Performs mean pooling over timesteps

        Args:
            last_hidden_state: hidden states of last layer          [B x L x E]
            attention_mask: masks to avoid [PAD] in computation     [B x L]

        Returns:
            x: sentence representation                              [B x E]

        """
        is_pad = (attention_mask == 0)  # [B x L]
        x = last_hidden_state.float().masked_fill(
            mask=is_pad.unsqueeze(-1),
            value=0.0
        ).type_as(last_hidden_state)                                # [B x L x E]
        x_lengths = attention_mask.sum(dim=-1, keepdims=True)       # [B x 1]
        x = torch.sum(x, dim=1)                                     # [B x E]
        x = torch.div(x, x_lengths).type_as(last_hidden_state)      # [B x E]
        return x


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        """ Performs max pooling over timesteps

        Args:
            last_hidden_state: hidden states of last layer          [B x L x E]
            attention_mask: masks to avoid [PAD] in computation     [B x L]

        Returns:
            x: sentence representation                              [B x E]

        """
        is_pad = (attention_mask == 0)                              # [B x L]
        x = last_hidden_state.float().masked_fill(
            mask=is_pad.unsqueeze(-1),
            value=float('-inf')
        ).type_as(last_hidden_state)                                # [B x L x E]
        x_lengths = attention_mask.sum(dim=-1, keepdims=True)       # [B x 1]
        x = torch.max(x, dim=1)                                     # [B x E]
        x = torch.div(x, x_lengths).type_as(last_hidden_state)      # [B x E]
        return x


class MeanMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_pool = MeanPool()
        self.max_pool = MaxPool()

    def forward(self, last_hidden_state, attention_mask):
        """ Performs mean-max pooling over timesteps

        Args:
            last_hidden_state: hidden states of last layer          [B x L x E]
            attention_mask: masks to avoid [PAD] in computation     [B x L]

        Returns:
            x: sentence representation                              [B x 2E]

        """
        x_mean = self.mean_pool(last_hidden_state, attention_mask)      # [B x E]
        x_max = self.max_pool(last_hidden_state, attention_mask)        # [B x E]
        x = torch.cat([x_mean, x_max], dim=1)                           # [B x 2E]
        return x
