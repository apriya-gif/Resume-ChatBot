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
        return history + [[question, "🤖 Model is still loading, please wait a moment..."]], ""
    
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
        
        history.append([question, answer])
        return history, ""

def submit_contact_form(name, email, message):
    if not name or not email or not message:
        return "❌ Please fill in all fields.", name, email, message
    
    # Log the contact request (in real deployment, you'd email this to Ameesha)
    contact_info = f"""
    New Contact Request:
    Name: {name}
    Email: {email}
    Message: {message}
    """
    print(contact_info)  # In production, send email instead
    
    return f"✅ Thank you {name}! Your message has been sent. Ameesha will reach out to you at {email} soon.", "", "", ""

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.header-section {
    background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
    padding: 2rem;
    margin-bottom: 2rem;
    border-radius: 12px;
    color: white;
    box-shadow: 0 10px 25px rgba(8, 145, 178, 0.2);
}

.quick-facts {
    background: rgba(15, 23, 42, 0.9);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.skill-tag {
    background: linear-gradient(135deg, #0891b2, #06b6d4);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    margin: 0.25rem;
    display: inline-block;
    font-size: 0.875rem;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(8, 145, 178, 0.3);
}

.chat-container {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 12px;
    border: 1px solid #334155;
    backdrop-filter: blur(10px);
}

.suggested-prompts button {
    background: rgba(8, 145, 178, 0.1) !important;
    border: 1px solid #0891b2 !important;
    color: #06b6d4 !important;
    border-radius: 20px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.875rem !important;
    margin: 0.25rem !important;
}

.suggested-prompts button:hover {
    background: rgba(8, 145, 178, 0.2) !important;
}
"""

with gr.Blocks(css=custom_css, title="Ameesha Priya - Resume Assistant") as iface:
    gr.HTML("""
    <div class="header-section">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 28px; font-weight: bold; border: 3px solid rgba(255,255,255,0.3);">
                    AP
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">Ameesha Priya</h1>
                    <p style="margin: 0; font-size: 1.1rem; opacity: 0.95; font-weight: 400;">Software Engineer – Backend, Distributed & FullStack Systems</p>
                </div>
            </div>
            <div>
                <a href="data/resume.txt" download="Ameesha_Priya_Resume.txt" style="background: rgba(255,255,255,0.2); color: white; padding: 0.75rem 1.5rem; border-radius: 25px; text-decoration: none; font-weight: 600; border: 2px solid rgba(255,255,255,0.3); transition: all 0.3s ease;">📄 Download Resume</a>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0; font-size: 1.2rem; font-weight: 600;">⚡ Quick Facts</h3>
                <p style="margin: 0.5rem 0;"><strong style="color: #e2e8f0;">Experience:</strong> <span style="color: #94a3b8;">4+ years</span></p>
                <p style="margin: 0.5rem 0;"><strong style="color: #e2e8f0;">Education:</strong> <span style="color: #94a3b8;">MS Software Engineering, CMU</span></p>
                <p style="margin: 0.5rem 0;"><strong style="color: #e2e8f0;">Specialization:</strong> <span style="color: #94a3b8;">Distributed Systems</span></p>
                <p style="margin: 0.5rem 0;"><strong style="color: #e2e8f0;">Recent Role:</strong> <span style="color: #94a3b8;">SDE at Bank of America</span></p>
            </div>
            """)
            
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0; font-size: 1.2rem; font-weight: 600;">🛠️ Core Skills</h3>
                <div style="margin-top: 1rem;">
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
                <h3 style="color: #06b6d4; margin-top: 0; font-size: 1.2rem; font-weight: 600;">🔗 Links</h3>
                <p style="margin: 0.5rem 0;">🔗 <a href="#" style="color: #06b6d4; text-decoration: none; font-weight: 500;">LinkedIn</a></p>
                <p style="margin: 0.5rem 0;">⭐ <a href="#" style="color: #06b6d4; text-decoration: none; font-weight: 500;">GitHub</a></p>
            </div>
            """)
            
            gr.HTML("""
            <div class="quick-facts">
                <h3 style="color: #06b6d4; margin-top: 0; font-size: 1.2rem; font-weight: 600;">📧 Contact Me</h3>
                <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">Interested in connecting? Fill out the form below and I'll reach out personally.</p>
            </div>
            """)
            
            contact_name = gr.Textbox(label="Your Name", placeholder="Enter your full name", container=True)
            contact_email = gr.Textbox(label="Your Email", placeholder="your.email@example.com", container=True)
            contact_message = gr.Textbox(
                label="Message", 
                placeholder="Tell me about the opportunity...",
                lines=3,
                container=True
            )
            contact_submit = gr.Button("Send Message", variant="primary", size="sm")
            contact_output = gr.Textbox(label="Status", interactive=False, visible=False)
        
        with gr.Column(scale=2):
            gr.HTML('<div class="chat-container" style="padding: 1.5rem;">')
            gr.Markdown("## 💬 Chat with Ameesha")
            gr.Markdown("Ask me anything about my experience, skills, or projects!")
            
            chatbot = gr.Chatbot(
                value=[["", "Hi! I'm Ameesha Priya's interactive resume assistant. I can tell you about my experience, skills, projects, education, and more. Feel free to ask me anything!"]],
                height=400,
                type="messages"
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
                <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.5rem;">💡 Try asking:</p>
            </div>
            """)
            
            with gr.Row(elem_classes="suggested-prompts"):
                exp_btn = gr.Button("Tell me about your experience", size="sm")
                proj_btn = gr.Button("What projects have you worked on?", size="sm")
                skills_btn = gr.Button("What are your technical skills?", size="sm")
                edu_btn = gr.Button("Tell me about your education", size="sm")
            
            gr.HTML('</div>')
    
    def handle_submit(message, history):
        return answer_question(message, history)
    
    def handle_contact_submit(name, email, message):
        return submit_contact_form(name, email, message)
    
    def use_suggested_prompt(prompt_text, history):
        return answer_question(prompt_text, history), ""
    
    submit_btn.click(handle_submit, [msg, chatbot], [chatbot, msg])
    msg.submit(handle_submit, [msg, chatbot], [chatbot, msg])
    
    # Suggested prompt handlers
    exp_btn.click(lambda h: use_suggested_prompt("Tell me about your experience", h), [chatbot], [chatbot, msg])
    proj_btn.click(lambda h: use_suggested_prompt("What projects have you worked on?", h), [chatbot], [chatbot, msg])
    skills_btn.click(lambda h: use_suggested_prompt("What are your technical skills?", h), [chatbot], [chatbot, msg])
    edu_btn.click(lambda h: use_suggested_prompt("Tell me about your education", h), [chatbot], [chatbot, msg])
    
    contact_submit.click(
        handle_contact_submit,
        [contact_name, contact_email, contact_message],
        [contact_output, contact_name, contact_email, contact_message]
    )
    
    contact_submit.click(lambda: gr.update(visible=True), outputs=contact_output)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
