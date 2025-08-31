import gradio as gr
import torch
import gc

# Clear GPU memory before loading the model
gc.collect()
torch.cuda.empty_cache()

from llama_query import retrieve, model, tokenizer, device

# Function to answer questions
def answer_question(question):
    with torch.no_grad():  # ensures no extra GPU memory is used
        return retrieve(question)

# Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response"),
    title="Resume ChatBot"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # creates a public link
    )
