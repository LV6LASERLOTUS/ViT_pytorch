import torch
import torch.nn as nn
from axial_rope import RoPE


class Attention(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()

        self.rope = RoPE(embed_size)
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

        # Perform RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # switch last 1 & 2 dimension of Kand multiply Q
        attention_score = Q @ K.transpose(-1, -2)

        # d_k as the dimension of Q and K
        attention_score = attention_score / (K.shape[-1] ** 0.5)

        # apply soft max to every row
        attention_weights = torch.softmax(attention_score, dim=-1) @ V

        return attention_weights


class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.final_projection = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        attention_score = torch.matmul(Q, K.transpose(-1, -2))

        # d_k as the dimension of Q and K
        attention_score = attention_score / (K.shape[-1] ** 0.5)

        # apply soft max to every row
        attention_weights = torch.softmax(attention_score, dim=-1) @ V

        return attention_weights

    def forward(self, x):
        """
        Split x into Q,K,V with embedding size d_head, perform attention on each Q,K,V
        concatenate the heads into multiheads.

        :param self: Description
        :param x: A tensor of shape (Batch, number of patches, embedding_size)
        """
        batch_size, num_patches, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split embeddings

        d_head = self.d_model // self.num_heads

        Q = Q.view(batch_size, num_patches, self.num_heads, d_head).transpose(1, 2)
        K = K.view(batch_size, num_patches, self.num_heads, d_head).transpose(1, 2)
        V = V.view(batch_size, num_patches, self.num_heads, d_head).transpose(1, 2)
        # Compute attention per head

        output_heads = self.scaled_dot_product_attention(Q, K, V)

        # Transpose to (batch_size, num_patches, num_heads, d_head) Concatenate all output head

        output = (
            output_heads.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_patches, self.d_model)
        )

        return self.final_projection(output)
