import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj_sparse):
        Wh = self.W(x)
        
        N = Wh.size(0)
        Wh_repeated = Wh.repeat(N, 1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0)
        all_combinations = torch.cat([Wh_repeated, Wh_repeated_alternating], dim=1)
        e = self.leakyrelu(self.a(all_combinations)).view(N, N) 
        
        e = e.to_sparse() * adj_sparse.coalesce()
        attention = torch.sparse.softmax(e, dim=1)
        
        h_prime = torch.sparse.mm(attention, Wh)
        return F.relu(h_prime)

class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)
    
    def forward(self, x):
        x = x.T
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
                
        att = torch.matmul(q, k.transpose(0,1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output.T
    
    