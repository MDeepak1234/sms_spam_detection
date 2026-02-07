import streamlit as st
import pickle 
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    s=[]
    for i in text:
        if i.isalnum():
            s.append(i)
    text=s[:]
    s.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            s.append(i)
    text=s[:]
    s.clear()
    for i in text:
        s.append(ps.stem(i))
    return " ".join(s)
tf=pickle.load(open('vectorizer.pkl','rb'))
m=pickle.load(open('model.pkl','rb'))
st.title("SMS Spam Classifier")
input_sms=st.text_area("Enter the message")
if st.button('Predict'):
    #preprocessing
    text_sms=text_transform(input_sms)
    #vectorizer
    vect_inp=tf.transform([text_sms])
    #model
    res=m.predict(vect_inp)[0]
    #output
    if res==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
