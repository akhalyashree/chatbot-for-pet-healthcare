import joblib
import random
import streamlit as st
import json

clf = joblib.load("chatbot.joblib")
vectorizer = joblib.load("vectorizer.joblib")

with open("intents.json", "r") as file:
    intents = json.load(file)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Chatbot")
    st.write("chatbot")
    
    user_input = st.text_input("You:")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

if __name__ == '__main__':
   main()