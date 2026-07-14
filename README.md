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
│   ├── 01_jax_basics.ipynb        # jit, vmap, PRNG
│   ├── 02_forward_process.ipynb   # noise schedules, closed-form forward jump
│   ├── 03_reverse_process.ipynb   # MLP denoiser, training loop
│   ├── 04_sampling.ipynb          # DDPM/DDIM samplers
│   ├── 05_cfg.ipynb               # classifier-free guidance
│   └── 06_dit.ipynb               # Diffusion Transformer (Flax NNX)
├── solutions/                     # reference implementations the notebooks import
└── slides.html                    # tutorial slides (open directly in a browser)
```

Each notebook has exercise cells marked `# YOUR CODE HERE`, followed by a collapsed **💡 Solution** cell you can reveal if you get stuck — try the exercise first. Code that's given rather than an exercise (schedules, previously-completed functions, etc.) is imported from `solutions/`, so each notebook stays self-contained regardless of where you start.

## For participants

**1. Get the materials.** Either open notebooks straight from GitHub in Colab (see below, no cloning needed), or clone the repo if you want a local copy:

```bash
git clone https://github.com/maigimenez/let-it-rip
cd let-it-rip
```

**2. Pick how you'll run the notebooks** — Colab (recommended) or local:

### Option A — Google Colab (recommended, zero setup)

No installation, and a free GPU. For each notebook, go to [colab.research.google.com](https://colab.research.google.com), choose **File → Open notebook → GitHub**, enter `maigimenez/let-it-rip`, and pick the notebook — or open the URL directly:

```
https://colab.research.google.com/github/maigimenez/let-it-rip/blob/main/notebooks/<name>.ipynb
```

Run the first cell in each notebook (labelled *Setup — run this cell first*) — it installs the few extra packages Colab doesn't ship with. Then, in Colab's menu, switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU**.

### Option B — Run locally

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter notebook
```

A GPU isn't required — everything also runs on CPU, just slower for training cells.

**3. Follow the session order.** The tutorial runs as two 90-minute sessions, each notebook picking up where the last left off (see `slides.html` for the full timing breakdown):

| Session | Notebook | Topic |
|---|---|---|
| 1 | `01_jax_basics` | `jit`, `grad`, `vmap`/`pmap`, PRNG |
| 1 | `02_forward_process` | noise schedules, closed-form forward jump |
| 1 | `03_reverse_process` | time embeddings, MLP denoiser, training loop |
| 2 | `04_sampling` | DDPM reverse step, DDIM, step-count trade-offs |
| 2 | `05_cfg` | class conditioning, classifier-free guidance |
| 2 | `06_dit` | patch embeddings, AdaLN, Flax NNX, full DiT |

## Prerequisites

- Comfortable with Python and NumPy
- Basic familiarity with neural networks (you don't need to know JAX)
- A Google account for Colab, or `uv` installed for a local run
- Internet access during the session — notebooks download CIFAR-10 from Hugging Face, and one bonus cell in `04_sampling` pulls a pretrained checkpoint

## About

Built by [Mai Giménez](https://ep2026.europython.eu/speaker/mai-gimenez), staff research engineer at Google DeepMind, for EuroPython 2026.
