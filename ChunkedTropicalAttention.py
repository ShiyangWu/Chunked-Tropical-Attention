import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch import nn
import torch




class TropicalLinear(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(TropicalLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(-2)
        
        W_expanded = self.W.unsqueeze(0)
        
        Wx = x_expanded + W_expanded  
        y, _ = torch.max(Wx, dim=-1)   
        return y

class TropicalAttention(nn.Module):
    def __init__(self, d_model, n_heads, device, tropical_proj=True, tropical_norm=False, symmetric=True):
        super(TropicalAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        
        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.tropical_proj:
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm:
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def normalize_tropical(self, x):
        return x - self.lambda_param

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        if self.tropical_norm:
            q = self.normalize_tropical(torch.log1p(F.relu(x)))
            k = self.normalize_tropical(torch.log1p(F.relu(x)))
            v = self.normalize_tropical(torch.log1p(F.relu(x)))
        else:
            q = torch.log1p(F.relu(x))
            k = torch.log1p(F.relu(x))
            v = torch.log1p(F.relu(x))
        
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        B = batch_size * self.n_heads
        q = q.reshape(B, seq_len, self.d_k)  # [B, S, D]
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        if self.symmetric:
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [B, S, S, D]
            max_diff, _ = diff.max(dim=-1)  # [B, S, S]
            min_diff, _ = diff.min(dim=-1)  # [B, S, S]
            d_trop = max_diff - min_diff    # [B, S, S]
            attn_scores = - d_trop           
        else:
            diff = q.unsqueeze(2) - k.unsqueeze(3)   # [B, S, 1, D] - [B, 1, S, D] -> [B, S, S, D]  
            sum_diff = diff.sum(dim=-1)              # [B, S, S]
            min_diff = diff.amin(dim=-1)             # [B, S, S]
            n = q.size(-1)
            attn_scores = - (sum_diff - n * min_diff)
        
        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B, S, S, D]
        context = sum_sv.max(dim=2).values  # [B, S, D]
        
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        context = torch.expm1(context)
        output = self.out(context)
        
        return output, attn_scores

# Chunked Tropical Attention
class ChunkedTropicalAttention(nn.Module):

    def __init__(self, d_model, n_heads, device, tropical_proj=True, tropical_norm=False):
        super(ChunkedTropicalAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.device = device
        
        if self.tropical_proj: 
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)
        
        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.tropical_norm:  
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))


    def normalize_tropical(self, x):
        return x - self.lambda_param


    def compute_attention_chunk(self, q, k, v, chunk_size):
        B, seq_len, d_k = q.size()
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        context = torch.full((B, seq_len, d_k), 
                        float('-inf'), 
                        device=self.device)
        
        for i in range(num_chunks):
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, seq_len)
            
            q_chunk = q[:, start_i:end_i, :]  # [B, chunk_size, D]
            
            attn_scores = self.compute_symmetric_attention_chunk(q_chunk, k)
            
            attn_expanded = attn_scores.unsqueeze(-1)  # [B, chunk_size, S, 1]
            v_expanded = v.unsqueeze(1)               # [B, 1, S, D]
            
            weighted_v = attn_expanded + v_expanded   # [B, chunk_size, S, D]
            
            context_chunk = weighted_v.max(dim=2)[0]  # [B, chunk_size, D]
            
            context[:, start_i:end_i, :] = context_chunk
        
        return context    

    def compute_symmetric_attention_chunk(self, q_chunk, k):
        B, chunk_size, D = q_chunk.shape
        S = k.shape[1]

        attn_scores = torch.zeros(B, chunk_size, S, device=self.device)    
        
        feature_chunk_size = min(80, D)  
        num_feature_chunks = (D + feature_chunk_size - 1) // feature_chunk_size
        
        for fc in range(num_feature_chunks):
            f_start = fc * feature_chunk_size
            f_end = min((fc + 1) * feature_chunk_size, D)
            
            q_sub = q_chunk[:, :, f_start:f_end]  # [B, chunk_size, feature_chunk]
            k_sub = k[:, :, f_start:f_end]        # [B, S, feature_chunk]
            
            diff = q_sub.unsqueeze(2) - k_sub.unsqueeze(1)  # [B, chunk_size, S, feature_chunk]
            
            if fc == 0:
                max_diff = diff.max(dim=-1)[0]
                min_diff = diff.min(dim=-1)[0]
            else:
                max_diff = torch.maximum(max_diff, diff.max(dim=-1)[0])
                min_diff = torch.minimum(min_diff, diff.min(dim=-1)[0])
        
        d_trop = max_diff - min_diff
        attn_scores = -d_trop
        
        return attn_scores
    
    
    def forward(self, x):       

        batch_size, seq_len, _ = x.size()
        
        if self.tropical_norm:   
            q = self.normalize_tropical(torch.log1p(F.relu(x)))
            k = self.normalize_tropical(torch.log1p(F.relu(x)))
            v = self.normalize_tropical(torch.log1p(F.relu(x)))
        else:
            q = torch.log1p(F.relu(x))
            k = torch.log1p(F.relu(x))
            v = torch.log1p(F.relu(x))
        
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        B = batch_size * self.n_heads
        q = q.reshape(B, seq_len, self.d_k)
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)
        
        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)
        
        context = self.compute_attention_chunk(q, k, v, chunk_size=80)   
        
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        context = torch.expm1(context)
        output = self.out(context)
        
        return output, None

