# PyMentor: Fine-Tuned Beginner Python Coding Assistant

## Goal
PyMentor is a QLoRA fine-tuned Python learning assistant for engineering students. It focuses on beginner-friendly explanations, code debugging, and basic programming exercises.

## Model
- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Fine-tuning method: QLoRA, 4-bit quantization
- Intended training hardware: 6 GB NVIDIA GPU

## Setup
1. Create and activate a virtual environment.
2. Install CUDA-enabled PyTorch for your NVIDIA GPU.
3. Run `pip install -r requirements.txt`.
4. Run `python src/prepare_dataset.py`.
5. Run `python src/train.py`.
6. Run `python app.py`.

## Dataset
The included 20 examples validate the pipeline only. Expand to 400-800 reviewed, varied examples before final training. Keep the test set held out.

## Evaluation
Compare the base and tuned model on identical unseen Python prompts. Score technical correctness, beginner clarity, helpfulness, and code quality. Report actual results and limitations.

## Limitations
The assistant may generate incorrect code. Users must run and test code independently.

## AI-tool disclosure
Document any AI tools used to create, review, debug, or improve this project, as required by the assignment.
