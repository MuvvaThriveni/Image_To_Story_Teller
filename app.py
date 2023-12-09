from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import os
from PIL import Image
import io
from langchain.llms import OpenAI
import streamlit as st
from gtts import gTTS
from dotenv import find_dotenv, load_dotenv 
load_dotenv(find_dotenv())

# Define the Hugging Face Hub API Token (Replace with your actual token)
HUGGINGFACEHUB_API_TOKEN = 'your api key'

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    You possess a remarkable talent for storytelling;
    weaving narratives that captivate and resonate. Let's craft a compelling short story with 3 small paragraphs:

    CONTEXT:{scenario}
    STORY:
    """

    # Corrected input variable name to 'input_variable' instead of 'input_variables'
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1, openai_api_key='Provide your api key'), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

def main():
    # Set Streamlit page config
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="✨",  # You can customize the icon
    )

    # Sidebar layout
    st.sidebar.title("✨ Upload Your Image! ✨")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    # Page layout
    st.markdown("<h2 style='text-align: center; margin-top: -30px;'>Transforming Images to Audio Story 📸➡️🔊</h2>", unsafe_allow_html=True)
    #st.subheader("Picture the Story, Hear the Magic! 🔮🎶")
    

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        # Calling Hugging Face model to create Description from img
        scenario = img2text(uploaded_file.name)
        # calling openai To generate Story
        story = generate_story(scenario)

        # Display scenario and story in a single column
        with st.expander("Scenario"):
            st.write(scenario)

        with st.expander("Story"):
            st.write(story)

        # Play the audio
        tts = gTTS(text=story, lang='en')  # You can specify the language if needed
        # Save the audio as audio.mp3
        tts.save('audio.mp3')
        audio_file = open('audio.mp3', 'rb')
        audio_bytes = audio_file.read()

        # Display audio player
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == '__main__':
    main()
