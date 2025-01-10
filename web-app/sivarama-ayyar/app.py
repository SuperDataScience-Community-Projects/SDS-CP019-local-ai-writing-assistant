import gradio as gr
import os
import requests
from huggingface_hub import InferenceClient
from bs4 import BeautifulSoup
import google.generativeai as genai

# A class to represent a Webpage

class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


system_message = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Do not use any logos. Respond in markdown."

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
genai.configure(api_key=GEMINI_API_KEY)


def stream_gemini(prompt):
    gemini = genai.GenerativeModel(
        model_name='gemini-1.5-flash-001',
        safety_settings=None,
        system_instruction=system_message
    )

    response = gemini.generate_content(prompt, safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}], stream=True)

    result = ""
    for chunk in response:
        result += chunk.text
        yield result


def stream_llama(prompt):
    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        stream=True
    )
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ''
        yield result


def stream_qwen(prompt):
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        stream=True
    )
    result = ""
    for chunk in response:
        result += chunk.choices[0].delta.content or ''
        yield result


def stream_brochure(company_name, url, model, language, response_tone):
    prompt = f"Please generate a {response_tone} company brochure in {language} for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model == "Llama":
        result = stream_llama(prompt)
    elif model == "Gemini":
        result = stream_gemini(prompt)
    elif model == "Qwen":
        result = stream_qwen(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


view = gr.Interface(
    fn=stream_brochure,
    inputs=[
        gr.Textbox(label="Company name:"),
        gr.Textbox(label="Landing page URL including http:// or https://"),
        gr.Dropdown(["Gemini", "Qwen", "Llama"], label="Select model"),
        gr.Dropdown(["English", "French", "Spanish", "German", "Hindi"]),
        gr.Dropdown(["Informational", "Promotional", "Humorous", "Business"], label="Select tone")],
    outputs=[gr.Markdown(label="Brochure:")],
    flagging_mode="never",
    title="Welcome to vsstech's Brochure Generator",
    description="Generates short brochure for company URL in selected language and tone."
)
view.launch()