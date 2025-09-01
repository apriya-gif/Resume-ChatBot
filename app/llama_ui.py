import gradio as gr
import torch
from llama_query import model, tokenizer, device, retrieve
import time

# -------------------
# Chatbot state
# -------------------
chat_history = []

# -------------------
# Function to handle chat with typing effect
# -------------------
def answer_question(user_input, history):
    history.append((user_input, ""))  
    yield history, history  
    
    time.sleep(0.5)
    context_chunks = retrieve(user_input, k=3)
    if not context_chunks:
        bot_answer = "I don't know."
    else:
        context_text = "\n".join(context_chunks)
        prompt = (
            f"You are an assistant answering questions about a resume.\n"
            f"Here is the resume content:\n{context_text}\n\n"
            f"Question: {user_input}\n"
            f"Answer:"
        )

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
        bot_answer = answer

    history[-1] = (user_input, bot_answer)
    yield history, history


# -------------------
# Gradio UI
# -------------------
with gr.Blocks(css="""
    .profile-card {
        background: #f9fafb;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .skills span {
        display: inline-block;
        background: #e5e7eb;
        color: #111827;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
    }
    .link-btn a {
        display: inline-block;
        margin-right: 12px;
        text-decoration: none;
        color: white;
        background: #2563eb;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 500;
    }
""") as demo:

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="profile-card">
                <h2 style="margin:0;">Ameesha Priya</h2>
                <p style="margin:2px 0; color:#4b5563;">Software Engineer – Backend, Distributed & FullStack Systems</p>
                <p style="margin:2px 0;">📧 apriya.gcp@gmail.com | 📱 (412) 499-6900</p>
            </div>
            <div class="profile-card">
                <h3>Quick Facts</h3>
                <ul style="margin:0; padding-left:16px; color:#374151;">
                    <li>Experience: 3+ years</li>
                    <li>Specialization: Distributed, Real-Time & Streaming Systems</li>
                    <li>Domain: Finance, Retail, E-commerce</li>
                </ul>
            </div>
            <div class="profile-card skills">
                <h3>Core Skills</h3>
                <span>Java</span><span>Spring Boot</span><span>Python</span><span>Distributed Systems</span><span>SQL</span><span>APIs</span>
            </div>
            <div class="profile-card link-btn">
                <a href="https://linkedin.com/in/ameesha-priya-2a773a136" target="_blank">LinkedIn</a>
                <a href="https://github.com/apriya-gif" target="_blank">GitHub</a>
            </div>
            """)

        with gr.Column(scale=2):
            gr.Markdown("## 💬 Resume Assistant")
            chatbot = gr.Chatbot()
            message = gr.Textbox(
                label="Ask me about Ameesha's resume", 
                placeholder="e.g., Tell me about your distributed systems experience"
            )
            
            with gr.Row():
                clear = gr.Button("Clear Chat")
            
            # Suggested questions
            with gr.Row():
                q1 = gr.Button("Tell me about your experience")
                q2 = gr.Button("What are your core skills?")
                q3 = gr.Button("What domains have you worked in?")

            # Events
            message.submit(answer_question, [message, chatbot], [chatbot, chatbot])
            clear.click(lambda: [], None, chatbot)
            q1.click(lambda _: "Tell me about your experience", None, message)
            q2.click(lambda _: "What are your core skills?", None, message)
            q3.click(lambda _: "What domains have you worked in?", None, message)

# -------------------
# Launch
# -------------------
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
