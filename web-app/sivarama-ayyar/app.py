import gradio as gr
from huggingface_hub import InferenceClient
import os
import requests
from bs4 import BeautifulSoup
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
 # A class to represent a Webpage

class Website:
    """
    A utility class to represent a Website that we have scraped
    """
    url: str
    title: str
    text: str

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# Define our system prompt
system_prompt = "You are a creative writing assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond with key points."

# A function that writes a User Prompt that asks for summaries of websites:

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; \
please provide a short, crisp summary of this website with key points. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def get_urls_from_prompt(prompt):
    urls=[]
    if prompt is not None:
        x= prompt.split()
        for i in x:
            if i.find("https:")==0 or i.find("http:")==0:
                urls.append(i)
    return urls

def summarizer_bot(prompt, history):
    url = get_urls_from_prompt(prompt)
    try:
       prompt += user_prompt_for(Website(url[0]))
    except:
       prompt += "No URL provided"

    messages = [{"role": "assistant", "content": system_prompt}] + history + [{"role": "user", "content": prompt}]

    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
    MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        stream=True
    )
    response = ""
    for chunk in stream:
        chunk_text = chunk.choices[0].delta.content
        response += chunk_text
        yield response

gr.ChatInterface(
    fn=summarizer_bot,
    type="messages",
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Type Question Here", container=False, scale=3),
    title="Summarizer bot",
    description="Give me the link to a website for which you want a summary of the contents of this website.",
    theme="Glass"
    ).launch()