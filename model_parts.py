
#model_parts.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
    def forward(self,x):
        mean_x=x.mean(dim=-1,keepdim=True)
        std_x=x.std(dim=-1,keepdim=True)
        ans=(self.alpha*(x-mean_x))/(std_x+1e-6)+self.beta
        return ans

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.ll1=nn.Linear(d_model,d_ff)
        self.drop=nn.Dropout(dropout)
        self.ll2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        return self.ll2(self.drop(F.gelu(self.ll1(x))))

class SelfAttention(nn.Module):
    def __init__(self,feature_dim,head_dim):
        super().__init__()
        self.w_q=nn.Linear(feature_dim,head_dim)
        self.w_k=nn.Linear(feature_dim,head_dim)
        self.w_v=nn.Linear(feature_dim,head_dim)
        
    def forward(self,x,mask=None):
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        d_k=k.size(-1)
        attn_score=q@k.transpose(-2, -1)/(d_k ** 0.5)
        if mask is not None:
            attn_score=attn_score.masked_fill(mask==0,float("-1e9"))
        attn_weight=F.softmax(attn_score,dim = -1)
        return attn_weight@v


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.h_dim=d_model//n_heads
        self.heads = nn.ModuleList([SelfAttention(d_model, self.h_dim) for _ in range(n_heads)])
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self,x,mask):
        outputs = [head(x,mask) for head in self.heads]
        x = torch.cat(outputs, dim=-1)
        return self.out_proj(x)
        


class Encoder(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,n_heads)
        self.norm1=LayerNorm(d_model)
        self.ff=FeedForward(d_model,d_model*4)
        self.norm2=LayerNorm(d_model)
    def forward(self,x,mask=None):
        temp=x
        x=self.attn(x,mask)
        temp=self.norm1(temp+x)
        x=self.ff(temp)
        x=self.norm2(temp+x)
        return x

class EncoderStack(nn.Module):
    def __init__(self, num_layers, d_model, n_heads):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(d_model, n_heads) for _ in range(num_layers)])
         
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x




    




