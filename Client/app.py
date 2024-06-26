import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model1 = load_model('Model/LSTMm1.keras')
# model2 = load_model('YT Search Model\LSTMm2.h5')

# Load tokenizer and max sequence length
with open('Model/tokenizer.pkl', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

# with open('YT Search Model\YTtokenizer.pkl', 'rb') as handle:
#     tokenizer2 = pickle.load(handle)

  # Adjust as per your training
# Function to make predictions
def predict_next_word(model, input_text, tokenizer, max_sequence_len , upto):
    for i in range(upto):
        token_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_token_text = pad_sequences([token_text],maxlen=max_sequence_len,padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word,index in tokenizer.word_index.items():
            if index == pos:
                input_text = input_text + " " + word
    return input_text

st.title('Next Word Predictor')
input_text = st.text_input('Enter the beginning of a sentence:', '')
upto = st.number_input('Enter the number of future predictions', min_value=1, max_value=20, value=5)
model_options = {
    "Professional Email Sentences Model": {
        "description": "This model is trained on professional email sentences such as 'I hope this email finds you well' and 'Thank you for your prompt response'. It generates phrases and completions suitable for professional communication.",
        "dataset_link": "https://www.kaggle.com/datasets/shorya22/next-word-predictor-dataset-nlp-task" 
    },
    "2024 Trending YouTube India Search Titles Model": {
        "description": "This model is trained on 2024 trending YouTube India search titles. It generates phrases and completions relevant to popular YouTube search queries in India for the year 2024.",
        "dataset_link": "https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset"
    }
}

st.title("Model Selection")
selected_model = st.selectbox("Select a model", options=list(model_options.keys()))
st.write("### You selected:", selected_model)
st.write("**Description:**", model_options[selected_model]["description"])
st.write("**Dataset Link:**", model_options[selected_model]["dataset_link"])

if selected_model == "Professional Email Sentences Model":
    model = model1
    tokenizer = tokenizer1
    max_sequence_len = 21
else:
    pass
    # model = model2
    # tokenizer = tokenizer2
    # max_sequence_len = 56

if st.button("Predict Next Words"):
    if input_text:
        next_word = predict_next_word(model, input_text, tokenizer, max_sequence_len , upto)
        st.write(f'Next word: {next_word}')