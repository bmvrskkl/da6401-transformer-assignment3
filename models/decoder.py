import torch.nn as nn
from models.attention import MultiHeadAttention
from models.encoder import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff,dropout):
        super().__init__()

        self.self_attention=MultiHeadAttention(d_model,heads)
        self.cross_attention=MultiHeadAttention(d_model,heads)

        self.ffn=FeedForward(d_model,d_ff,dropout)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        attn1,_=self.self_attention(x,x,x,tgt_mask)

        x=self.norm1(x+self.dropout(attn1))

        attn2,attention=self.cross_attention(
            x,
            enc_output,
            enc_output,
            src_mask
        )

        x=self.norm2(x+self.dropout(attn2))

        ffn_output=self.ffn(x)

        x=self.norm3(x+self.dropout(ffn_output))

        return x,attention
