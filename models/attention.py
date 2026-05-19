import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):

        batch_size, seq_len, d_model = x.size()

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size, num_heads, seq_len, d_k = x.size()

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V,
        mask=None
    ):

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        if mask is not None:

            # FORCE BOOLEAN
            mask = mask.bool()

            # expand dimensions safely
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)

            # APPLY BEFORE SOFTMAX
            scores = scores.masked_fill(
                ~mask,
                float('-1e9')
            )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        return output

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output = self.scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        output = self.combine_heads(
            attention_output
        )

        output = self.W_o(output)

        return output