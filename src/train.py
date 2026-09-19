import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Chosen for a 6 GB NVIDIA GPU. Do not commit downloaded weights or outputs to Git.
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUTPUT_DIR = "outputs/pymentor-0.5b-qlora"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Install a CUDA-enabled PyTorch build before training.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization, device_map="auto"
    )
    model.config.use_cache = False
    data = load_dataset("json", data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/validation.jsonl",
    })
    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
    data = data.map(format_chat)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"
    )
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=768,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=5,
        bf16=True,
        fp16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=data["train"],
        eval_dataset=data["validation"], peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

if __name__ == "__main__":
    main()
