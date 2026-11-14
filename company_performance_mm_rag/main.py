# from conversation import response_generation

# user_prompt = "What is this company about?"
# response, context = response_generation(user_prompt,[])

# print(f"\nUser: {user_prompt}\n")
# print(f"\nAI: {response}\n")
# print(f"\nCONTEXT: {context}\n")

import gradio as gr
from conversation import response_generation

# # Import your custom backend function
# try:
#     from conversation import response_generation
# except ImportError:
#     # Fallback stub in case 'conversation.py' isn't in your local path yet
#     def response_generation(prompt, history):
#         return f"Echo response to: {prompt}", "Sample context retrieved from documents or database regarding the query."

def chat_interface(message, history):
    # Format history as expected by typical backends, or pass as is
    response, context = response_generation(message, history)
    return response, context

# Custom CSS for Deep Navy Blue Background and Clean Split Layout
custom_css = """
body, .gradio-container {
    background-color: #0B192C !important;
    color: #F0F4F8 !important;
}
.sidebar-panel, .chat-panel {
    background-color: #1E3E62 !important;
    border-radius: 12px !important;
    border: 1px solid #000000 !important;
    padding: 15px !important;
}
h1, h2, h3, label, span, p {
    color: #F0F4F8 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 NovaCore AI Chatbot")
    gr.Markdown("Conversational AI Demonstration by Sidhant Pati")
    
    with gr.Row():
        # Left Column: Custom Chatbot & Input Panel
        with gr.Column(scale=1, elem_classes="chat-panel"):
            gr.Markdown("### 💬 Conversation")
            chatbot_ui = gr.Chatbot(height=450)
            msg_input = gr.Textbox(placeholder="Ask something about the company...", container=False)
            submit_btn = gr.Button("Send", variant="primary")
            
        # Right Column: Context Showcase Panel
        with gr.Column(scale=1, elem_classes="sidebar-panel"):
            gr.Markdown("### 📄 Retrieved Context")
            context_display = gr.Textbox(
                label="Live Context Stream",
                value="Context will appear here once you start chatting...",
                lines=20,
                max_lines=25,
                interactive=False
            )

    def handle_user_submission(user_message, history):
        if not user_message.strip():
            return history, "", ""
            
        # Call your backend function
        response, context = response_generation(user_message, history)
        
        # Ensure history is initialized
        if history is None:
            history = []
            
        # Append messages using the Gradio 6.0 dictionary format (role + content)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})
        
        return history, "", context

    # Wire up the submission events (Enter key or Send button)
    submittables = [msg_input, chatbot_ui]
    
    msg_input.submit(
        fn=handle_user_submission, 
        inputs=submittables, 
        outputs=[chatbot_ui, msg_input, context_display]
    )
    submit_btn.click(
        fn=handle_user_submission, 
        inputs=submittables, 
        outputs=[chatbot_ui, msg_input, context_display]
    )

if __name__ == "__main__":
    # Pass theme and css parameters to launch() in Gradio 6.0
    demo.launch(css=custom_css, theme=gr.themes.Soft())