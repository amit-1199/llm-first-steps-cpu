# LLM First Steps (CPU) — by @Amit-AI

I tried LLaMA first and hit issues on my laptop. So I learned the basics with a tiny model (distilgpt2) on CPU:
1) Check if you have GPU or CPU
2) Download → load → prompt → response

**Open in Colab:**

[![check_gpu_cpu](https://colab.research.google.com/drive/1623tOrqXijmmRNu6rfmBU69j7xQHHDcn?usp=sharing)](https://colab.research.google.com/github/amit-1199/llm-first-steps-cpu/blob/main/check_gpu_cpu.ipynb)
[![first llm run](https://colab.research.google.com/drive/138fHfUJywKL7VcFV7pKD5USGayKYvCce?usp=sharing)](https://colab.research.google.com/github/amit-1199/llm-first-steps-cpu/blob/main/distilgpt2.ipynb)

**Files**
- [`check_gpu_cpu.py`](https://github.com/amit-1199/llm-first-steps-cpu/blob/main/check_gpu_cpu.py)
- [`distilgpt2_quickstart.py`](https://github.com/amit-1199/llm-first-steps-cpu/blob/main/distilgpt2_quickstart.py)

**Screenshot**

![Terminal output showing CUDA False and a short model response](/img-terminal.png)


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
