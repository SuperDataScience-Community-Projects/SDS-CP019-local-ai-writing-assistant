import os
import requests
import gradio as gr
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

class Website:
    """
    A utility class to represent the website to scrape
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"

        # removing irrelevant information
        if soup.body:
            for irr in soup.body(["script","style","img","input"]):
                irr.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
def get_url_from_prompt(prompt):

    words = [word for word in prompt.split(" ")]
    url = [word for word in words if 'http' in word]

    return url

def brochure_bot(prompt, history):

    system_prompt = "You are a creative writing assistant"
    system_prompt += " and your job is to help write a brochure for the company that the user provides."
    system_prompt += " Make sure to keep your responses professional and formal."
    system_prompt += " If you do not know information to a question the user provides,"
    system_prompt += " mention clearly that you do not know the answer."
    system_prompt += " Your first task is to ask the user to give you the URL of a website."
    system_prompt += " If you do not find one, ask the user for one."
    system_prompt += " Do not ask the user for any additional input, just keep the tone formal."

    url = get_url_from_prompt(prompt)
    try:
        website_info = Website(url[0]).text
        prompt += website_info
    except:
        prompt += "No URL provided"

    messages = [
        {"role":"assistant", "content":system_prompt},
        {"role":"user", "content":prompt}
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

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Welcome to the AI Brochure Creator
        This app will use the URL link to a website of your choice and create a commercial brochure for the company.
        In the below box, enter the URL of the website you wish to create a brochure for.
        
        """
    )
    url = gr.Textbox(label="Website URL")
    output = gr.Textbox(label="Output")
    submit = gr.Button("Submit")
    submit.click(fn=brochure_bot, inputs=url, outputs=output)

demo.launch()