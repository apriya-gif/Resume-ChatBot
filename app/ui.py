import threading
import gradio as gr
import torch
import gc

# Clear GPU memory
gc.collect()
torch.cuda.empty_cache()

model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    from llama_query import retrieve, model as m, tokenizer as t, device as d
    model, tokenizer, device = m, t, d
    print("Model loaded!")

# Load model in background thread
threading.Thread(target=load_model).start()

def answer_question(question):
    if model is None:
        return "Model is still loading, please wait..."
    import llama_query
    with torch.no_grad():
        return llama_query.retrieve(question)

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response"),
    title="Resume ChatBot"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
