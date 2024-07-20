import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Paths to the model and tokenizer
model_path = 'Model/LSTMm1.keras'
tokenizer_path = 'Model/tokenizer.pkl'

# Check if the model and tokenizer paths exist
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    try:
        model1 = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer1 = pickle.load(handle)
            
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
else:
    st.error("Model or tokenizer files not found. Please check the file paths.")

def predict_next_word(model, input_text, tokenizer, max_sequence_len, upto):
    for i in range(upto):
        token_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_sequence_len, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                input_text = input_text + " " + word
    return input_text

st.title('Next Word Predictor Using LSTM')
input_text = st.text_input('Enter the beginning of a sentence:', '')
upto = st.number_input('Enter the number of future predictions', min_value=1, max_value=20, value=5)
st.write("**Description:** This model is trained on professional email sentences such as 'I hope this email finds you well' and 'Thank you for your prompt response'. It generates phrases and completions suitable for professional communication.")
st.write("**Dataset Link:** https://www.kaggle.com/datasets/shorya22/next-word-predictor-dataset-nlp-task")

if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    model = model1
    tokenizer = tokenizer1
    max_sequence_len = 21

    if st.button("Predict Next Words"):
        if input_text:
            next_word = predict_next_word(model, input_text, tokenizer, max_sequence_len, upto)
            st.write(f'Next words: {next_word}')
else:
    st.error("Model or tokenizer files not found. Please ensure the files are in the correct directory.")
