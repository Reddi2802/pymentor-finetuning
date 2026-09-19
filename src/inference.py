import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = "outputs/pymentor-0.5b-qlora/final_adapter"
SYSTEM_PROMPT = "You are PyMentor, a supportive Python tutor for engineering students who are beginners. Explain clearly and use short, correct examples."

def load_tuned_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer

def answer(model, tokenizer, question):
    messages = [{"role":"system", "content":SYSTEM_PROMPT}, {"role":"user", "content":question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
