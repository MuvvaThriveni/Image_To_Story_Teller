from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import os
from langchain.llms import OpenAI
import streamlit as st
from gtts import gTTS
load_dotenv(find_dotenv())


# Define the Hugging Face Hub API   Token (Replace with your actual token)
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """
    you are a story teller;
    You can generate a short story based on a simple narrative, the story should be more than 20 words;

    CONTEXT:{scenario}
    STORY:
    """

    # Corrected input variable name to 'input_variable' instead of 'input_variables'
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1, openai_api_key='provide_your_own_APIkey'), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

# text to speech



def main():
    # Set Streamlit page config
    st.set_page_config(
    page_title="Image to Audio Story",
    page_icon="✨",  # You can customize the icon
    layout="centered",)

    # Page layout
    st.title("Image to Audio Story 📸➡️🔊")
    st.subheader("Turn an Image into an Audio Story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        #text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        # Play the audio
        tts = gTTS(text=story, lang='en')  # You can specify the language if needed
        # Save the audio as audio.mp3
        tts.save('audio.mp3')
        audio_file = open('audio.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
 
if __name__ == '__main__':
    main()
