# llm-first-steps-cpu
# amit-AI
# LLM First Steps (CPU) — by @Amit-AI

I tried LLaMA first and hit issues on my laptop. So I learned the basics with a tiny model (distilgpt2) on CPU:
1) Check if you have GPU or CPU
2) Download → load → prompt → response

**Open in Colab:**  
[![Open In Colab]
(https://colab.research.google.com/assets/colab-badge.svg)]
(
https://colab.research.google.com/github/Amit-AI/llm-first-steps-cpu/blob/main/notebooks/check_gpu_cpu.ipynb
https://colab.research.google.com/github/Amit-AI/llm-first-steps-cpu/blob/main/notebooks/distilgpt2.ipynb
)


## Files
- `check_gpu_cpu.py` — prints if CUDA/GPU is available (name, VRAM), else CPU.
- `distilgpt2_quickstart.py` — minimal Transformers pipeline on CPU.

## How to run (local)
```bash
pip install -U transformers accelerate torch
python check_gpu_cpu.py
python distilgpt2_quickstart.py

Notes
CPU is fine for learning the workflow.
If slow, keep max_new_tokens small (e.g., 80).
I used ChatGPT, GitHub Copilot, and Google Gemini to write/debug the code.

---
