import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch import nn
import time
from ChunkedTropicalAttention import TropicalAttention, ChunkedTropicalAttention



def measure_peak_memory_and_time(attention_module, seq_len, batch_size=1, dim=80):

    x = torch.randn(batch_size, seq_len, dim).cuda()
    
    _ = attention_module(x)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    _ = attention_module(x)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    return peak_memory, elapsed_time_ms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
### Tropical Attenttion
attn = TropicalAttention(    
            d_model=80, 
            n_heads=1, 
            device=device,
            tropical_proj=True,
            tropical_norm=False
).to(device)

'''

### Chunked Tropical Attention
attn = ChunkedTropicalAttention(    
            d_model=80, 
            n_heads=1, 
            device=device,
            tropical_proj=True,
            tropical_norm=False
).to(device)




for seq_len in [512, 1024, 2048, 4096, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16000, 32000, 48000, 50000]:
    mem, t = measure_peak_memory_and_time(attn, seq_len)
    print(f"Seq Len: {seq_len}, Peak Memory: {mem:.2f} GB, Time: {t:.2f} ms")

