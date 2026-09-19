import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = "outputs/pymentor-0.5b-qlora/final_adapter"
TEST_PATH = "data/processed/test.jsonl"
RESULTS_DIR = Path("results")

SYSTEM_PROMPT = (
    "You are PyMentor, a supportive Python tutor for engineering students "
    "who are beginners. Explain clearly and use short, correct examples."
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant,
        device_map="auto"
    )

    tuned = PeftModel.from_pretrained(base, ADAPTER_PATH)
    tuned.eval()

    return tuned, tokenizer


def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Use 10 test questions to keep evaluation manageable.
    test_data = test_data[:10]

    model, tokenizer = load_model()

    results = []

    for index, example in enumerate(test_data, start=1):
        question = next(
            message["content"]
            for message in example["messages"]
            if message["role"] == "user"
        )

        print(f"Evaluating question {index}/{len(test_data)}...")

        with model.disable_adapter():
            base_answer = generate_answer(model, tokenizer, question)

        tuned_answer = generate_answer(model, tokenizer, question)

        results.append({
            "question": question,
            "base_model_response": base_answer,
            "fine_tuned_model_response": tuned_answer
        })

    with open(
        RESULTS_DIR / "comparison_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(
        RESULTS_DIR / "comparison_examples.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("# Base Model vs Fine-Tuned Model Comparison\n\n")
        f.write("Model: Qwen2.5-Coder-0.5B-Instruct\n\n")
        f.write("Evaluation set: 10 held-out test questions\n\n")

        for index, result in enumerate(results, start=1):
            f.write(f"## Question {index}\n\n")
            f.write(f"**Question:** {result['question']}\n\n")
            f.write("### Base Model Response\n\n")
            f.write(result["base_model_response"] + "\n\n")
            f.write("### Fine-Tuned Model Response\n\n")
            f.write(result["fine_tuned_model_response"] + "\n\n")
            f.write("---\n\n")

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "evaluation_questions": len(results),
            "status": "Responses generated. Manual scoring required.",
            "metrics": {
                "technical_correctness": "Pending",
                "beginner_clarity": "Pending",
                "helpfulness": "Pending",
                "code_quality": "Pending"
            }
        }, f, indent=2)

    print("\nEvaluation complete!")
    print("Results saved to results/")


if __name__ == "__main__":
    main()