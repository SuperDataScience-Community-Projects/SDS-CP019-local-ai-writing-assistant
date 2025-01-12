import random
import os
import gradio as gr
import google.generativeai as genai  

# Initialize the Gemini model
Model = genai.GenerativeModel('gemini-1.5-flash')  

def chat_short_story(length, genre, theme, tone, writing_style):
    """
    Generates a creative short story using Gemini API.
    Parameters:
        length (int): Approximate word count for the story.
        genre (str): Genre of the story.
        theme (str): Central theme of the story.
        tone (str): Tone of the story.
        writing_style (str): Writing style for the story.
    Returns:
        str: The generated short story, or an error message if unsuccessful.
    """
    # Retrieve the Gemini API key from the environment
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not found. Please set the GEMINI_API_KEY environment variable."

    # Configure the Gemini API
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"Error: Failed to configure the Gemini API. Details: {e}"

    # Construct the prompt
    prompt = (
        f"Write a creative short story of approximately {length} words in the {genre} genre. "
        f"Use a {writing_style} writing style with a {tone} tone. "
        f"The story should revolve around the theme of {theme}. "
        f"Ensure it is engaging and includes a suitable title."
    )

    # Generate a response using the Gemini model
    try:
        chat = Model.start_chat()  
        response = chat.send_message(prompt)  
        return response.text  
    except Exception as e:
        return f"Error: Failed to generate story. Details: {e}"


# Predefined options
Length = [100, 250, 750]
r_length = random.choice([l for l in Length if l >= 100])

Genre = [
    "Fiction", "Nonfiction", "Drama", "Poetry", "Fantasy", "Horror", "Mystery",
    "Science Fiction", "Suspense", "Women's fiction", "Supernatural/Paranormal", "Young adult"
]
r_genre = random.choice(Genre)

Themes = [
    "Love", "Redemption", "Forgiveness", "Coming of age", "Revenge", "Good vs evil",
    "Bravery and hardship", "The power of social status", "The destructive nature of love",
    "The fallibility of the human condition"
]
r_themes = random.choice(Themes)

Writing_Styles = ["Expository", "Narrative", "Descriptive", "Persuasive", "Creative"]
r_Style = random.choice(Writing_Styles)

Tones = ["Formal", "Optimistic", "Worried", "Friendly", "Curious", "Assertive", "Encouraging"]
r_tones = random.choice(Tones)

# Gradio Interface setup
iface = gr.Interface(
    fn=chat_short_story,
    inputs=[
        gr.Slider(value=100, label="Story Length", minimum=100, maximum=2500, step=50),
        gr.Dropdown(label="Story Genre", choices=Genre),
        gr.Dropdown(label="Story Theme", choices=Themes),
        gr.Dropdown(label="Writing Style", choices=Writing_Styles),
        gr.Dropdown(label="Story Tone", choices=Tones)
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="Welcome to Patrick's Story Generator",
    description="Generate creative short stories tailored to your preferences."
)

iface.launch()
