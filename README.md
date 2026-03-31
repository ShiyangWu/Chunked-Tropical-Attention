# Chunked Tropical Attention (CTA)

This repository contains the official PyTorch implementation of **Chunked Tropical Attention (CTA)**.


## 📌 Overview

Tropical attention offers a powerful paradigm for algorithmic reasoning but suffers from quadratic memory complexity, limiting its application to long sequences. **Chunked Tropical Attention (CTA)** decomposes tropical metric computation along both sequence and feature dimensions, reducing peak memory from \(O(N^2)\) to \(O(C \times N)\) while preserving the original time complexity and the algebraic properties of tropical geometry.

This repository provides the core implementation of CTA, which can be used as a drop‑in replacement for standard attention modules in any neural network.

## 📁 File Structure

```
.
├── ChunkedTropicalAttention.py   # Main implementation of CTA
├── AbalationExperiments.py       # Script to reproduce memory/time ablation results
└── README.md                     # This file
```

## 🔧 Dependencies

- Python 3.8+
- PyTorch 1.12+
- CUDA (recommended for GPU testing)

Install dependencies:
```bash
pip install torch
```

## 🚀 Usage

### 1. Import and instantiate CTA

```python
import torch
from ChunkedTropicalAttention import ChunkedTropicalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create CTA module
attn = ChunkedTropicalAttention(
    d_model=80,           # feature dimension
    n_heads=1,            # number of attention heads
    device=device,
    tropical_proj=True,   # enable tropical linear projection
    tropical_norm=False   # disable tropical normalization
).to(device)
```

### 2. Forward pass

```python
batch_size = 1
seq_len = 16000
dim = 80

x = torch.randn(batch_size, seq_len, dim).to(device)
output, _ = attn(x)   # output shape: [batch_size, seq_len, d_model]
```

### 3. Reproduce ablation experiments

Run the ablation script to measure peak GPU memory and forward time for different sequence lengths:

```bash
python AbalationExperiments.py
```

Expected output (example):
```
Seq Len: 512, Peak Memory: 0.04 GB, Time: 5.67 ms
Seq Len: 1024, Peak Memory: 0.27 GB, Time: 9.74 ms
...
Seq Len: 16000, Peak Memory: 8.56 GB, Time: 709.85 ms
```

You can modify the sequence length list and module type inside the script.

### 4. Integrate CTA into your own model

Simply replace your existing attention module with `ChunkedTropicalAttention`. The interface is compatible with typical Transformer‑like layers.

```python
self.attention = ChunkedTropicalAttention(d_model=512, n_heads=8, device=device)
```

## 📊 Key Results

| Sequence Length | Original Tropical Attention (OTA) | CTA (this work) |
|-----------------|-----------------------------------|-----------------|
| 512             | 0.18 GB / 1.07 ms                 | 0.04 GB / 5.67 ms |
| 8,192           | 43.95 GB / 199.21 ms              | 2.69 GB / 210.30 ms |
| 16,000          | OOM                               | 8.56 GB / 709.85 ms |

CTA achieves up to **99.5% memory reduction** while maintaining comparable time complexity.

## 📝 License

This code is provided for research purposes. It is the intellectual property of **Foshan University**. Please contact the authors for licensing inquiries.


## 🤝 Acknowledgements

This work builds upon the foundational contributions of:
- Zhang et al., *Tropical Geometry of Deep Neural Networks* (ICML 2018)
- Hashemi et al., *Tropical Attention* (NeurIPS 2025)

---

**For any questions, please contact:** wusy@fosu.edu.cn