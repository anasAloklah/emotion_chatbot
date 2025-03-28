import streamlit as st
from transformers import pipeline
import google.generativeai as genai
import json
import random

# Load language configurations from JSON
with open('languages_config.json', 'r', encoding='utf-8') as f:
    LANGUAGES = json.load(f)['LANGUAGES']

# Load the JSON data for emotion templates
with open('emotion_templates.json', 'r') as f:
    data = json.load(f)
    
# Configure Gemini (replace with your API key)
genai.configure(api_key="AIzaSyCYRYNwCU1f9cgJYn8pd86Xcf6hiSMwJr0")
model = genai.GenerativeModel('gemini-2.0-flash') 

def generate_text(prompt, context=""):
    """
    Generates text using the Gemini model.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating text: {e}")
        return "I am sorry, I encountered an error while generating the text."

def create_prompt(emotion, topic=None):
    """
    Chooses a random prompt from the template list.
    """
    templates = data["emotion_templates"][emotion]
    prompt = random.choice(templates)
    if topic:
        # Replace various placeholders in the prompt
        placeholders = ["[topic/person]", "[topic]", "[person]", "[object]", "[outcome]"]
        for placeholder in placeholders:
            prompt = prompt.replace(placeholder, topic)
    
    subfix_prompt = "Make the generated text in the same language as the topic.\n"
    subfix_prompt += "Make the generated text short.\n"
    
    prefix_prompt = "## topic\n" + topic
    prompt = subfix_prompt + prompt + prefix_prompt
    return prompt
    
# 1. Emotion Detection Model (Using Hugging Face's transformer)
emotion_classifier = pipeline("text-classification", model="AnasAlokla/multilingual_go_emotions")

# 2. Conversational Agent Logic
def get_ai_response(user_input, emotion_predictions):
    """Generates AI response based on user input and detected emotions."""
    dominant_emotion = None
    max_score = 0
    responses = None
    for prediction in emotion_predictions:
        if prediction['score'] > max_score:
            max_score = prediction['score']
            dominant_emotion = prediction['label']
    
    prompt_text = create_prompt(dominant_emotion, user_input)
    responses = generate_text(prompt_text)
    
    # Handle cases where no specific emotion is clear
    if dominant_emotion is None:
        return "Error for response"
    else:
        return responses

# 3. Streamlit Frontend
def main():
    # Language Selection
    selected_language = st.sidebar.selectbox(
        "Select Interface Language",
        list(LANGUAGES.keys()),
        index=0  # Default to English
    )
    
    # Display Image
    st.image('chatBot_image.jpg', channels='RGB')
    
    # Set page title and header based on selected language
    st.title(LANGUAGES[selected_language]['title'])
    
    # Input Text Box
    user_input = st.text_input(
        LANGUAGES[selected_language]['input_placeholder'], 
        ""
    )
    
    if user_input:
        # Emotion Detection
        emotion_predictions = emotion_classifier(user_input)
        
        # Display Emotions
        st.subheader(LANGUAGES[selected_language]['emotions_header'])
        for prediction in emotion_predictions:
            st.write(f"- {prediction['label']}: {prediction['score']:.2f}")
        
        # Get AI Response
        ai_response = get_ai_response(user_input, emotion_predictions)
        
        # Display AI Response
        st.subheader(LANGUAGES[selected_language]['response_header'])
        st.write(ai_response)

# Run the main function
if __name__ == "__main__":
    main()