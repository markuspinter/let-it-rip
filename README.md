# Let It Rip 🎨 — A Diffusion Tutorial

Materials for the [EuroPython 2026](https://ep2026.europython.eu/session/let-it-rip-a-diffusion-tutorial) tutorial on building a Denoising Diffusion Probabilistic Model (DDPM) from scratch using JAX.

> **Tuesday, 14 July 2026 · 09:30–12:45 · Conference Hall Complex B**

## What you'll build

A DDPM backed by a Diffusion Transformer (DiT) — the same architecture behind state-of-the-art text-to-image models — implemented from the ground up in JAX.

Along the way you'll learn:

- **JAX fundamentals** — `jit`, `vmap`, and explicit PRNG handling
- **The forward process** — noise schedules and SDEs
- **The reverse process** — score matching and denoising
- **Sampling** — DDPM and DDIM samplers
- **Classifier-free guidance (CFG)** — steering generation with a conditioning signal
- **Diffusion Transformers** — patch embeddings, self-attention, and AdaLN conditioning

## Structure

```
let-it-rip/
├── notebooks/
│   ├── 01_jax_basics.ipynb        # JIT, vmap, PRNG
│   ├── 02_forward_process.ipynb   # noise schedules, SDEs
│   ├── 03_reverse_process.ipynb   # score matching, denoising
│   ├── 04_sampling.ipynb          # DDPM/DDIM samplers
│   ├── 05_cfg.ipynb               # classifier-free guidance
│   └── 06_full_ddpm.ipynb         # putting it all together
├── ddpm/                          # simplified implementations (fill these in!)
│   ├── schedule.py
│   ├── dit.py
│   ├── diffusion.py
│   ├── sampling.py
│   └── guidance.py
├── solutions/                     # complete reference implementations
└── slides/
```

The notebooks have exercises with blanks to fill in. `ddpm/` is what you build during the tutorial; `solutions/` has the full implementations including a complete DiT with AdaLN and class conditioning.

## Running the notebooks

All notebooks are designed to run on **Google Colab** — no local GPU setup needed. Each notebook has a setup cell at the top that installs everything.

If you'd like to run locally:

```bash
git clone https://github.com/maigimenez/let-it-rip
cd let-it-rip
uv sync
uv run jupyter notebook
```

## Prerequisites

- Comfortable with Python and NumPy
- Basic familiarity with neural networks (you don't need to know JAX)
- A Google account for Colab

## About

Built by [Mai Giménez](https://ep2026.europython.eu/speaker/mai-gimenez), staff research engineer at Google DeepMind, for EuroPython 2026.
