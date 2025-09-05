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

def answer_question(question, history):
    if not model_ready:
        return history + [("", "Model is still loading, please wait...")], ""
    
    import llama_query
    with torch.no_grad():
        # Get relevant chunks
        chunks = llama_query.retrieve(question)
        
        # Build context and generate response (same as original llama_query.py)
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
        
        inputs = llama_query.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llama_query.device)
        
        with torch.no_grad():
            outputs = llama_query.model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llama_query.tokenizer.eos_token_id
            )
        
        response = llama_query.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        
        # Clean up the response
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        
        history.append((question, answer))
        return history, ""

def submit_contact_form(name, email, message):
    if not name or not email or not message:
        return "Please fill in all fields."
    
    # Log the contact request (in real deployment, you'd email this to Ameesha)
    contact_info = f"""
    New Contact Request:
    Name: {name}
    Email: {email}
    Message: {message}
    """
    print(contact_info)  # In production, send email instead
    
    return f"Thank you {name}! Your message has been sent. Ameesha will reach out to you at {email} soon."

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', sans-serif;
}

.header-section {
    background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
    padding: 2rem;
    margin-bottom: 2rem;
    border-radius: 12px;
    color: white;
}

.quick-facts {
    background: rgba(15, 23, 42, 0.8);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 1rem;
}

.skill-tag {
    background: #0891b2;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin: 0.25rem;
    display: inline-block;
    font-size: 0.875rem;
}

.chat-container {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    border: 1px solid #334155;
}
"""

with gr.Blocks(css=custom_css, title="Ameesha Priya - Resume Assistant") as iface:
    gr.HTML("""
    <div class="header-section">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="width: 60px; height: 60px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold;">
                    AP
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 2rem; font-weight: bold;">Ameesha Priya</h1>
                    <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">Software Engineer – Backend, Distributed & FullStack Systems</p>
                </div>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0;">Quick Facts</h3>
                <p><strong>Experience:</strong> 4+ years</p>
                <p><strong>Education:</strong> MS Software Engineering, CMU</p>
                <p><strong>Specialization:</strong> Distributed Systems</p>
                <p><strong>Recent Role:</strong> SDE at Bank of America</p>
            </div>
            """)
            
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0;">Core Skills</h3>
                <div>
                    <span class="skill-tag">Java</span>
                    <span class="skill-tag">Spring Boot</span>
                    <span class="skill-tag">Kafka</span>
                    <span class="skill-tag">AWS</span>
                    <span class="skill-tag">Kubernetes</span>
                    <span class="skill-tag">Python</span>
                </div>
            </div>
            """)
            
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0;">Links</h3>
                <p>🔗 <a href="#" style="color: #06b6d4;">LinkedIn</a></p>
                <p>⭐ <a href="#" style="color: #06b6d4;">GitHub</a></p>
            </div>
            """)
        
        with gr.Column(scale=2):
            gr.HTML('<div class="chat-container">')
            gr.Markdown("## 💬 Chat with Ameesha")
            gr.Markdown("Ask me anything about my experience, skills, or projects!")
            
            chatbot = gr.Chatbot(
                value=[("", "Hi! I'm Ameesha Priya's interactive resume assistant. I can tell you about my experience, skills, projects, education, and more. Feel free to ask me anything!")],
                height=400
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything about my background...",
                    container=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            gr.HTML("""
            <div style="margin-top: 1rem;">
                <p style="color: #94a3b8; font-size: 0.875rem;">Try asking:</p>
            </div>
            """)
            
            with gr.Row():
                gr.Button("Tell me about your experience", size="sm")
                gr.Button("What projects have you worked on?", size="sm")
                gr.Button("What are your technical skills?", size="sm")
                gr.Button("Tell me about your education", size="sm")
            
            gr.HTML('</div>')
    
    with gr.Tab("📧 Contact Ameesha"):
        gr.Markdown("## Get in Touch")
        gr.Markdown("Interested in connecting? Fill out the form below and Ameesha will reach out to you personally.")
        
        with gr.Row():
            with gr.Column():
                contact_name = gr.Textbox(label="Your Name", placeholder="Enter your full name")
                contact_email = gr.Textbox(label="Your Email", placeholder="your.email@example.com")
                contact_message = gr.Textbox(
                    label="Message", 
                    placeholder="Tell me about the opportunity or how you'd like to connect...",
                    lines=5
                )
                contact_submit = gr.Button("Send Message", variant="primary")
                contact_output = gr.Textbox(label="Status", interactive=False)
    
    def handle_submit(message, history):
        return answer_question(message, history)
    
    submit_btn.click(handle_submit, [msg, chatbot], [chatbot, msg])
    msg.submit(handle_submit, [msg, chatbot], [chatbot, msg])
    
    contact_submit.click(
        submit_contact_form,
        [contact_name, contact_email, contact_message],
        contact_output
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
