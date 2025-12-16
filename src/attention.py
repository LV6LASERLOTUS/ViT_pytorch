import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """Docstring for forward

        :param self: Description
        :param x: A tensor of shape (Batch, number of patches, embedding_size)
        """

        # Compute Q,K,V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # switch last 1 & 2 dimension of Kand multiply Q
        attention_score = Q @ K.transpose(-1, -2)

        # d_k as the dimension of Q and K
        attention_score = attention_score / (K.shape[-1] ** 0.5)

        # apply soft max to every row
        attention_weights = torch.softmax(attention_score, dim=-1) @ V

        return attention_weights
