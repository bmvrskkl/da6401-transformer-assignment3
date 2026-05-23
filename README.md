# DA6401 Assignment 3 — Transformer for Neural Machine Translation

Implementation of the Transformer architecture from scratch using PyTorch, including Multi-Head Attention, Positional Encoding, Noam Scheduler, Attention Visualization, and extensive ablation studies based on the paper *Attention Is All You Need*.

## Links

GitHub: https://github.com/bmvrskkl/da6401-transformer-assignment3

W&B Report: https://wandb.ai/bskkl04-indian-institute-of-technology-madras/da6401_transformer/reports/TRANSFORMER-FOR-MACHINE-TRANSLATION--VmlldzoxNjkzODEzNg?accessToken=c9h7lra46fypddvyn8ci3y6s2ybgcsvymdge9vtst7skni7bc7czmi9txbvq6uk4

---

# Project Structure

```text
DA6401_Full_Starter/
├── README.md
├── config.py
├── utils.py
├── train.py
├── inference.py
├── attention_visualization.py
├── transformer.pth
├── models/
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── positional_encoding.py
│   └── scheduler.py
├── attention_heads/
│   ├── head_0.png
│   ├── head_1.png
│   ├── head_2.png
│   ├── ...
│
└── wandb/
