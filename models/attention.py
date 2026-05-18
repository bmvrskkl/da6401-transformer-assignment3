import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V,mask=None):
        d_k=Q.size(-1)

        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        attention=torch.softmax(scores,dim=-1)

        output=torch.matmul(attention,V)

        return output,attention


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        self.fc_out=nn.Linear(d_model,d_model)

        self.attention=ScaledDotProductAttention()

    def split_heads(self,x,batch_size):
        x=x.view(batch_size,-1,self.num_heads,self.d_k)
        return x.transpose(1,2)

    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)

        Q=self.W_q(query)
        K=self.W_k(key)
        V=self.W_v(value)

        Q=self.split_heads(Q,batch_size)
        K=self.split_heads(K,batch_size)
        V=self.split_heads(V,batch_size)

        output,attention=self.attention(Q,K,V,mask)

        output=output.transpose(1,2).contiguous()

        output=output.view(batch_size,-1,self.d_model)

        output=self.fc_out(output)

        return output,attention
