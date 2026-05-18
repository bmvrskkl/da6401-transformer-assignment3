import torch.nn as nn
from models.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )

    def forward(self,x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff,dropout):
        super().__init__()

        self.attention=MultiHeadAttention(d_model,heads)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.ffn=FeedForward(d_model,d_ff,dropout)

        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output,_=self.attention(x,x,x,mask)

        x=self.norm1(x+self.dropout(attn_output))

        ffn_output=self.ffn(x)

        x=self.norm2(x+self.dropout(ffn_output))

        return x
