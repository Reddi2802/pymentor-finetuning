# PyMentor Evaluation Report

## Evaluation setup

- Base model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Fine-tuned model: base model plus the trained QLoRA adapter
- Held-out questions: 10
- Generation mode: deterministic
- Maximum new tokens: 64
- Scoring scale: 1–5

## Why the response time varies

The model is running locally on an RTX 4050 Laptop GPU with 6 GB VRAM.
QLoRA reduces memory consumption, but it does not make generation
instantaneous.

1. **Autoregressive generation:** The model generates one token at a time.
   Longer responses require more sequential computation.

2. **First-run overhead:** The first generation may include CUDA
   initialization, memory allocation, and kernel setup.

3. **Model and adapter overhead:** The quantized base model and LoRA adapter
   must occupy GPU memory, along with activations and the key-value cache.

4. **Limited VRAM:** With 6 GB VRAM, memory allocation and GPU scheduling
   can affect latency.

5. **Quantization behavior:** 4-bit quantization primarily reduces memory
   usage. It does not guarantee a proportional speed increase because some
   operations may involve dequantization or specialized kernels.

6. **Windows and available kernels:** Some optimized acceleration libraries
   may be less consistently available on Windows. The training logs also
   showed that Triton was unavailable; this warning does not necessarily
   indicate a model failure.

7. **Different prompt and output lengths:** Similar questions can still
   produce different numbers of tokens, causing different timings.

## Why we retained `max_new_tokens=64`

Increasing the limit to 128 or 256 could make answers more complete, but it
also increases the maximum possible generation time. The project targets
short beginner explanations, so 64 tokens is acceptable for the current
demo.

The trade-off is that some responses were truncated. This is documented as
an evaluation limitation rather than hidden.

## Scoring criteria

| Criterion | Meaning |
|---|---|
| Technical correctness | Factual accuracy |
| Beginner clarity | Ease of understanding |
| Helpfulness | Directness and usefulness |
| Code quality | Readability and suitability of examples |

Scores use a 1–5 scale:

- 1: Poor
- 2: Needs major improvement
- 3: Acceptable
- 4: Good
- 5: Excellent

## Preliminary summary

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Technical correctness | 3.4 | 4.9 |
| Beginner clarity | 2.8 | 4.5 |
| Helpfulness | 2.5 | 4.4 |
| Code quality | 1.9 | 4.4 |
| Overall average | 2.65 | 4.55 |

These scores are manually assigned and preliminary because several responses
were truncated. They should not be presented as a formal statistical
benchmark.

## Main observations

The fine-tuned model generally produced answers that were:

- More concise
- More directly aligned with the question
- More suitable for beginners
- More likely to include a short, complete example
- More consistent in tutoring style

The evaluation also identified an issue in the duplicate-removal answer:
`dict.fromkeys()` requires hashable list elements, so the explanation should
be improved.

## Recommended future evaluation

- Use a larger generation limit for a separate evaluation run.
- Increase the number of held-out questions.
- Use multiple evaluators.
- Test generated code automatically.
- Record latency, tokens per second, and peak GPU memory.
