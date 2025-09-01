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
    yield history, history  # immediate update to show user message
    
    time.sleep(0.5)  # typing delay
    
    # Retrieve relevant context
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
# Gradio UI with WhatsApp-style bubbles
# -------------------
with gr.Blocks(css="""
    .chatbox {max-height: 600px; overflow-y: auto;}
    .user-message {
        background-color: #DCF8C6;
        border-radius: 15px 15px 0px 15px;
        padding: 10px;
        margin: 5px;
        display: inline-block;
        text-align: left;
        float: right;
        clear: both;
        max-width: 70%;
    }
    .bot-message {
        background-color: #EAEAEA;
        border-radius: 15px 15px 15px 0px;
        padding: 10px;
        margin: 5px;
        display: inline-block;
        text-align: left;
        float: left;
        clear: both;
        max-width: 70%;
    }
""") as demo:

    gr.Markdown("## 📝 Resume ChatBot 💬")
    chatbot = gr.Chatbot(elem_classes=["chatbox"])

    message = gr.Textbox(
        label="Ask a question about your resume",
        placeholder="Type your question here and press Enter"
    )
    clear = gr.Button("Clear Chat")

    def render_chat(history):
        rendered = []
        for user, bot in history:
            rendered.append((f'<div class="user-message">{user}</div>', 
                             f'<div class="bot-message">{bot}</div>'))
        return rendered

    message.submit(answer_question, [message, chatbot], [chatbot, chatbot]).then(
        render_chat, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot)

# -------------------
# Launch
# -------------------
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
