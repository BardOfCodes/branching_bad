import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()
        self.num_heads = nh
        self.hidden_dim = hd

        self.self_attn = nn.MultiheadAttention(self.hidden_dim, self.num_heads)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, src, attn_mask, key_padding_mask):
        
        src = src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask,
            need_weights=False
        )[0]

        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(self.n2(src)), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)
        return src.transpose(0, 1)
        
class LearnablePositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout, max_len=256):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)
        pos_arange = torch.arange(max_len).unsqueeze(0)
        self.register_buffer("pos_arange", pos_arange)
        

    def forward(self, x):
        pe = self.pe(self.pos_arange.repeat(x.shape[0], 1))
        x = x + pe[:, : x.size(1)]
        return self.dropout(x)
    