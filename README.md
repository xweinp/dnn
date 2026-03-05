# Deep Neural Networks

Solutions for the Deep Neural Networks course at the University of Warsaw, 2025/2026.

Course page [here](https://mim-ml-teaching.github.io/public-dnn-2025-26/).

## Assignments

| # | Topic | What's inside |
|---|-------|---------------|
| [dnn1](dnn1/) | Image classification | Custom dataloading, augmentation, training loop, metrics — the whole PyTorch pipeline from scratch |
| [dnn2](dnn2/) | Image segmentation | Grad-CAM saliency maps, SAM-based segmentation pipelines (foreground/background point selection) |
| [dnn3](dnn3/) | Transformers from scratch | Attention, RoPE, SwiGLU, Mixture of Experts — building a transformer piece by piece |
| [dnn4](dnn4/) | RL | Exploration strategies and some experiments |

Each folder has its own README with run instructions and Colab links.

## Quick start

Most assignments use [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
cd dnn4
uv sync
```

Or just open the Colab links in the individual READMEs — no setup needed.