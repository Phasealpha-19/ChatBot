import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ssl


nltk.download('punkt_tab') 

# Load responses from the text file
def load_responses(file_path):
    """
    Load prompts and responses from a text file into a dictionary.
    Each line should be in the format: "prompt=response".
    """
    responses = {}
    try:
        with open('your_text_file.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if '=' in line:
                    prompt, response = line.strip().split('=', 1)
                    # Preprocess prompt to remove special characters
                    prompt_clean = preprocess_text(prompt)
                    responses[prompt_clean] = response
    except FileNotFoundError:
        st.error("The response file is missing. Please provide a valid file.")
    return responses

# Preprocess text by removing special characters and converting to lowercase
def preprocess_text(text):
    """
    Remove special characters and convert text to lowercase.
    """
    return re.sub(r'\W+', ' ', text).strip().lower()

# Get chatbot response
def get_response(user_input, responses):
    """
    Fetch the response for a user input from the responses dictionary.
    """
    user_input_clean = preprocess_text(user_input)
    return responses.get(user_input_clean, "Sorry, I don't have a response for that.")

# Streamlit app
def main():
    st.title("Custom Chatbot")
    st.write("Ask me something!")

    # Load the text file containing responses
    response_file = "your_text_file.txt"
    responses = load_responses(response_file)

    # Input for user query
    user_query = st.text_input("Your question:")

    if user_query:
        response = get_response(user_query, responses)
        st.write("**Chatbot:**", response)

if __name__ == "__main__":
    main()
