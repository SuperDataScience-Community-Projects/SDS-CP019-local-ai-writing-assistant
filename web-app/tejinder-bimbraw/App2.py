import os
import requests
import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub import InferenceClient

client = InferenceClient(token="HUGGINGFACE_API_KEY")

# Chat function for generating LinkedIn posts
def chat_llama2(prompt, files):
    system_message = "You are a professional LinkedIn post creator. Create a professional LinkedIn post based on the user's inputs."

    # Prepare the input prompt, including file names if provided
    file_info = ""
    images = []  # Collect image file paths to display in the output
    if files:
        for file in files:
            file_info += f"File uploaded: {file.name}\n"
            # Check if the file is an image
            if file.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                # Save the image to a temporary folder for display
                file_path = os.path.join("temp_images", file.name)
                os.makedirs("temp_images", exist_ok=True)  # Ensure the directory exists
                with open(file_path, "wb") as f:
                    f.write(file.encode("latin1"))  # Convert the string to bytes before writing
                images.append(file_path)

    user_prompt = f"{prompt}\n\n{file_info}"

    # Initialize the conversation history
    history = [
        {"role": "assistant", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Add uploaded images to the conversation history for display
    for image_path in images:
        history.append({"role": "user", "content": f'<img src="{image_path}" alt="Uploaded Image" style="max-width: 300px;">'})
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

    # Call the Ollama API
    result = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct", 
        messages=history, 
        max_tokens=500,
        stream=True
    )

    # Stream the response and update the history
    for chunk in result:
        if "message" in chunk and "content" in chunk["message"]:
            chunk_text = chunk["message"]["content"]
            if history[-1]["role"] == "assistant":
                # Append to the last assistant message
                history[-1]["content"] += chunk_text
            else:
                # Create a new assistant message
                history.append({"role": "assistant", "content": chunk_text})
            yield history  # Yield the updated history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LinkedIn Post Generator")
    gr.Markdown("Enter the topic and optionally upload files to generate a professional LinkedIn post.")

    chatbot = gr.Chatbot(elem_id="chatbot", height=400, type="messages")

    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Type the topic for your LinkedIn post here...",
            label="Post Topic",
            interactive=True
        )
        file_input = gr.File(
            file_types=["image", "application/pdf", "text"],
            label="Upload Files (Optional)",
            file_count="multiple"
        )

    submit_button = gr.Button("Generate Post")

    # Bind inputs and outputs
    submit_button.click(
        fn=chat_llama2,
        inputs=[text_input, file_input],
        outputs=chatbot
    )

# Launch the Gradio app
demo.launch()