# 3K Platform Demo

A **standalone demo** that showcases the 3K platform's functionality. No backend, API, or Kubernetes required—just open the HTML file in a browser.

## What It Shows

The demo presents the key capabilities of the 3K platform:

- **Overview** – 3 core metrics (1000+ GPUs, 100B+ params, 1000+ hours) and feature cards
- **Model Repository** – Browse models (LLaMA2, Qwen, ChatGLM, etc.) with Fine-tune/Deploy actions
- **Training Jobs** – Submit and monitor GPU training jobs (PyTorchJob, MPIJob)
- **Inference** – Deployed models and endpoints (KServe)
- **JupyterLab** – Development environments with GPU
- **Datasets** – Manage datasets for training
- **Cluster Info** – GPU nodes and availability
- **Playground** – Chat UI (display only, no actual inference)

All data is **mock/static**—buttons and links are for display only.

## How to Run

### Option 1: Open directly

```bash
# From project root
open examples/3k-platform-demo/index.html

# Or with a browser
xdg-open examples/3k-platform-demo/index.html   # Linux
```

### Option 2: Simple HTTP server (recommended)

```bash
cd examples/3k-platform-demo
python3 -m http.server 8080
# Then open http://localhost:8080
```

Or with Node.js:

```bash
cd examples/3k-platform-demo
npx serve .
# Then open the URL shown (e.g. http://localhost:3000)
```

## Files

- `index.html` – Main demo page
- `styles.css` – Styling
- `demo.js` – Mock data and navigation
- `README.md` – This file

## Notes

- This is a **UI demo only**. No API calls, no Kubernetes, no real training or inference.
- Use it for presentations, onboarding, or to preview the platform’s look and feel.
