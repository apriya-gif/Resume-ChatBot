import threading
import torch
import gc
import gradio as gr
from llama_query import model as llama_model, tokenizer as llama_tokenizer, device as llama_device, retrieve as llama_retrieve

gc.collect()
torch.cuda.empty_cache()

# ------------------- Model loading -------------------
model_ready = False
model = llama_model
tokenizer = llama_tokenizer
device = llama_device

def load_model():
    global model_ready
    # model is already imported from llama_query
    model_ready = True
    print("Model loaded!")

threading.Thread(target=load_model).start()

# ------------------- Chat functions -------------------
def answer_question(question, history):
    if not model_ready:
        history.append({"role": "assistant", "content": "🤖 Model is still loading, please wait a moment..."})
        return history, ""
    
    chunks = llama_retrieve(question)
    context = "\n".join(chunks)
    
    prompt = f"""You are Ameesha Priya's AI assistant. Answer questions about her resume and background.

IMPORTANT PRIVACY RULES:
- NEVER share phone numbers or personal contact information
- If asked for contact info, say "Please use the contact form to reach out"
- Stay focused on professional topics only
- Don't make up information not in the resume

Context from resume:
{context}

Question: {question}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""

def submit_contact_form(name, email, message):
    if not name or not email or not message:
        return "❌ Please fill in all fields.", name, email, message
    contact_info = f"""
New Contact Request:
Name: {name}
Email: {email}
Message: {message}
"""
    print(contact_info)
    return f"✅ Thank you {name}! Your message has been sent. Ameesha will reach out to you at {email} soon.", "", "", ""

# ------------------- Gradio UI -------------------
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""

with gr.Blocks(css=custom_css, title="Ameesha Priya - Resume Assistant") as iface:
    
    gr.HTML("<h1 style='color:white;'>Ameesha Priya - Interactive Resume Assistant</h1>")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Chat with Ameesha")
            gr.Markdown("Ask me anything about my experience, skills, or projects!")
            
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hi! I'm Ameesha Priya's interactive resume assistant. I can tell you about my experience, skills, projects, education, and more. Feel free to ask me anything!"}],
                height=400,
                type="messages"
            )
            
            msg = gr.Textbox(placeholder="Ask me anything about my background...", container=False, scale=4)
            submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            # Suggested prompts
            with gr.Row(elem_classes="suggested-prompts"):
                exp_btn = gr.Button("Tell me about your experience", size="sm")
                proj_btn = gr.Button("What projects have you worked on?", size="sm")
                skills_btn = gr.Button("What are your technical skills?", size="sm")
                edu_btn = gr.Button("Tell me about your education", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("## 📬 Contact Me")
            contact_name = gr.Textbox(label="Name")
            contact_email = gr.Textbox(label="Email")
            contact_message = gr.Textbox(label="Message", lines=4)
            contact_submit = gr.Button("Send Message", variant="primary")
            contact_output = gr.Markdown("")
    
    # ------------------- Event bindings -------------------
    submit_btn.click(answer_question, [msg, chatbot], [chatbot, msg])
    msg.submit(answer_question, [msg, chatbot], [chatbot, msg])
    
    exp_btn.click(lambda h: answer_question("Tell me about your experience", h), [chatbot], [chatbot, msg])
    proj_btn.click(lambda h: answer_question("What projects have you worked on?", h), [chatbot], [chatbot, msg])
    skills_btn.click(lambda h: answer_question("What are your technical skills?", h), [chatbot], [chatbot, msg])
    edu_btn.click(lambda h: answer_question("Tell me about your education", h), [chatbot], [chatbot, msg])
    
    contact_submit.click(submit_contact_form, [contact_name, contact_email, contact_message], [contact_output, contact_name, contact_email, contact_message])

# ------------------- Launch -------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
