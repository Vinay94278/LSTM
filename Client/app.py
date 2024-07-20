import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model = load_model('Model/LSTMm1.h5')

# Load tokenizer and max sequence length
with open('Model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, input_text, tokenizer, max_sequence_len , upto):
    for i in range(upto):
        token_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_token_text = pad_sequences([token_text],maxlen=max_sequence_len,padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word,index in tokenizer.word_index.items():
            if index == pos:
                input_text = input_text + " " + word
    return input_text

st.title('Next Word Predictor Using LSTM')
input_text = st.text_input('Enter the beginning of a sentence:', '')
upto = st.number_input('Enter the number of future predictions', min_value=1, max_value=20, value=5)
st.write("**Description:** This model is trained on professional email sentences such as 'I hope this email finds you well' and 'Thank you for your prompt response'. It generates phrases and completions suitable for professional communication.")
st.write("**Dataset Link:** https://www.kaggle.com/datasets/shorya22/next-word-predictor-dataset-nlp-task")

max_sequence_len = 21

if st.button("Predict Next Words"):
    if input_text:
        next_word = predict_next_word(model, input_text, tokenizer, max_sequence_len , upto)
        st.write(f'Next word: {next_word}')