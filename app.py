import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
}

textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Sidebar
with st.sidebar:
    st.header("📌 About")
    st.write("""
**Developer:** Sarthak Nigam

**Model:** Multinomial Naive Bayes

**Vectorizer:** TF-IDF

Built with **Python**, **NLTK**, **Scikit-learn**, and **Streamlit**.
""")

# Main page
st.title("📩 SMS Spam Detector")

st.write(
    "Enter an SMS message below to check whether it is **Spam** or **Ham**."
)

input_sms = st.text_area("✉️ Enter your message")

# Prediction
if st.button("🚀 Predict"):

    with st.spinner("Analyzing message..."):

        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message Detected")
    else:
        st.success("✅ This is a Ham Message")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Sarthak Nigam")
