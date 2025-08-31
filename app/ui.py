import gradio as gr
from llama_query import retrieve, model, tokenizer, device

def answer_question(query):
    # Retrieve context
    context_chunks = retrieve(query, k=3)
    if not context_chunks:
        return "I don't know."
    
    context_text = "\n".join(context_chunks)
    prompt = f"You are an assistant answering questions about a resume.\nHere is the resume content:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

# Create Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything about the resume..."),
    outputs="text",
    title="Resume ChatBot"
)

iface.launch(share=True)
