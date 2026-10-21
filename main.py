import gradio as gr

# A dummy function representing your LLM chatbot backend
def predict(message, history):
    # message: the latest user input string
    # history: previous conversation turns
    response = f"You said: {message}. Here is my LLM response!"
    return response

# Create the chat interface
demo = gr.ChatInterface(
    fn=predict, 
    title="My Local LLM Chatbot",
    textbox=gr.Textbox(
        placeholder="Type your message here...", 
        container=False, 
        scale=7
        )
)

if __name__ == "__main__":
    demo.launch()