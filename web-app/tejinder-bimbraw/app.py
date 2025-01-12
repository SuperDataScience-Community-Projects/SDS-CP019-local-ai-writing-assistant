import os
import requests
import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub import InferenceClient

client = InferenceClient(token="HUGGINGFACE_API_KEY")

def chat_llama2(prompt, history):
    system_message = "You are a professional LinkedIn post creator"
    
    messages = [
        {"role":"assistant","content":system_message},
        {"role":"user","content":prompt}
    ]


    
   
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct", 
        messages=messages, 
        max_tokens=500,
        stream=True
    )
    
    # streaming the response from the model
    # this prints out each output token form the model to have the resulting text displayed in real time
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content
    yield response



gr.ChatInterface(
    fn=chat_llama2 ,
    type="messages",
    chatbot=gr.Chatbot(type= 'messages',height=500),
    textbox=gr.Textbox(placeholder="Type Question Here", container=True, scale=3),
    title="LinkedIn Post Creator",
    description="What topic would you like to create the post about",
    theme="default",
    examples=["Latest project that your team delivered", "A new Learning and Development course you created", "Your personal achievement"]
    # cache_examples=True
    ).launch()



 