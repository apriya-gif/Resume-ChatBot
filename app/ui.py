import threading
import gradio as gr
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

model_ready = False
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device, model_ready
    from llama_query import model as m, tokenizer as t, device as d
    model, tokenizer, device = m, t, d
    model_ready = True
    print("Model loaded!")

threading.Thread(target=load_model).start()

def answer_question(question):
    if not model_ready:
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
