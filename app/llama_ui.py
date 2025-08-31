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
    # Append user message first
    history.append((user_input, ""))  
    yield history, history  # immediate update to show user message
    
    # Simulate bot typing
    time.sleep(0.5)
    
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

        # Tokenize & generate
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

    # Update the last chat with bot response
    history[-1] = (user_input, bot_answer)
    yield history, history

# -------------------
# Gradio UI
# -------------------
with gr.Blocks(css="""
    .chatbot-message {
        border-radius: 15px;
        padding: 10px;
        margin: 5px;
        max-width: 70%;
    }
    .user-message { background-color: #DCF8C6; text-align: right; }
    .bot-message { background-color: #F1F0F0; text-align: left; }
""") as demo:

    gr.Markdown("## 📝 Resume ChatBot 💬")
    chatbot = gr.Chatbot(elem_classes=["chatbot-message"])
    message = gr.Textbox(label="Ask a question about your resume", placeholder="Type your question here and press Enter")
    clear = gr.Button("Clear Chat")

    message.submit(answer_question, [message, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(share=True)
