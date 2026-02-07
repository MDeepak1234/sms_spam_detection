import streamlit as st
import pickle
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    words = []
    for i in text:
        if i.isalnum():
            words.append(i)

    filtered = []
    for i in words:
        if i not in stopwords.words('english') and i not in string.punctuation:
            filtered.append(i)

    stemmed = []
    for i in filtered:
        stemmed.append(ps.stem(i))

    return " ".join(stemmed)


tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = text_transform(input_sms)
        vector_input = tf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
