1. 下载并预处理所有的 AI cuda level 1 Kernels（预处理将注释掉所有非 device code）

```bash
/torch/venv3/pytorch/bin/python ./PreprocessCUKernels.py
```

2. 将 cuda kernels 批量转换为 BANG code

```bash
bash gen_bang_code.sh
```

---

```
@article{lange2025aicudaengineer,
    title     = {The AI CUDA Engineer: Agentic CUDA Kernel Discovery,
                 Optimization and Composition},
    author    = {Lange, Robert Tjarko and
                 Prasad, Aaditya and 
                 Sun, Qi and
                 Faldor, Maxence and
                 Tang, Yujin and
                 Ha, David},
    journal   = {arXiv preprint},
    year      = {2025}
}
```