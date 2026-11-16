# 🧠 Building a Large Language Model (LLM) From Scratch

Implementation of transformer-based autoregressive language models, GPT architecture, and token decoding strategies built from foundational mathematical principles in PyTorch.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 🔬 Core Topics & Architecture

### 1. Tokenization & Embedding Pipelines
- Byte-Pair Encoding (BPE) and subword tokenization mechanics.
- Token and positional embedding projections into latent vector space.

### 2. Multi-Head Attention Mechanism
- Scaled Dot-Product Self-Attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- Causal masking to enforce unidirectional autoregressive generation.
- Layer normalization (LayerNorm) and residual skip connections.

### 3. Autoregressive Decoding Strategies
- **Greedy Search**: Deterministic argmax token selection.
- **Temperature Scaling**: Softmax logit redistribution for controlling output entropy.
- **Top-K Sampling**: Truncating probability mass to the top $k$ candidate tokens.
- **Nucleus (Top-P) Sampling**: Dynamic cumulative probability distribution cutoffs.

---

## 📂 Repository Contents

```
Building-llm-from-scratch/
└── notebooks/
    ├── Copy_of_LLM_Decoding_Strategies.ipynb  # End-to-end tokenization & decoding notebook
    ├── gpt_download3.py                       # Pretrained checkpoint weight loader
    ├── loss-plot.pdf                          # Empirical cross-entropy training loss curves
    └── temperature-plot.pdf                   # Softmax temperature ablation visualizations
```

---

<sub>Academic implementations by [Kaushal Sharma](https://github.com/Kaushal-1925)</sub>
