import gradio as gr
from src.inference import load_tuned_model, answer

model, tokenizer = load_tuned_model()

def respond(question):
    if not question.strip():
        return "Please enter a Python question."
    return answer(model, tokenizer, question)

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=5, label="Ask a beginner Python question"),
    outputs=gr.Markdown(label="PyMentor response"),
    title="PyMentor",
    description="A QLoRA fine-tuned Beginner Python Coding Assistant for engineering students.",
    examples=["Why does range(5) stop at 4?", "Fix: for i in range(5) print(i)", "What is the difference between = and == in Python?"],
)

if __name__ == "__main__":
    demo.launch()
