# PyMentor

PyMentor is a beginner-focused Python tutor created by fine-tuning
`Qwen/Qwen2.5-Coder-0.5B-Instruct` with QLoRA.

## Project objective

The project adapts a general coding model to provide short, clear,
beginner-friendly Python explanations and examples for engineering students.

## Model and training configuration

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Parameters | Approximately 0.5B |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| Dataset size | 300 examples |
| Train/validation/test split | 240/30/30 |
| Epochs | 3 |
| Training steps | 90 |
| Training loss | Approximately 0.4038 |
| Evaluation loss | Approximately 0.09986 |
| Training duration | Approximately 26 minutes |
| Adapter | `outputs/pymentor-0.5b-qlora/final_adapter` |

## Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i5-13450HX |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| RAM | 16 GB |
| Operating system | Windows |

QLoRA was used because full fine-tuning would require considerably more
GPU memory. QLoRA loads the base model in 4-bit precision, freezes most
base-model parameters, and trains only small LoRA adapter matrices.

## Running the application

```bash
python app.py
```

The application launches a local Gradio interface.

## Evaluation

Ten held-out questions were used to compare the base model with the
same model plus the trained LoRA adapter. The evaluation considered
technical correctness, beginner clarity, helpfulness, and code quality.

The evaluation used deterministic generation with:

```python
max_new_tokens=64
```

The results are preliminary because some responses were truncated.

See `docs/evaluation.md` for the detailed report.

## Limitations

- Only 300 training examples were used.
- Only 10 test questions were manually evaluated.
- Some responses were truncated by the 64-token limit.
- Manual scores are preliminary.
- The duplicate-removal example needs a more precise explanation.
- Local latency depends on GPU memory, CUDA initialization, model loading,
  quantization kernels, prompt length, output length, and background tasks.

## Future improvements

- Expand and diversify the dataset.
- Evaluate more questions.
- Add automated tests for generated Python code.
- Measure tokens per second and peak VRAM usage.
- Add response streaming.
- Add a FastAPI backend and Docker deployment.
