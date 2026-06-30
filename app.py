import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- NLTK --------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# -------------------- CSS --------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(to bottom right,#eef5ff,#dbeafe);
}

/* Title */
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1d4ed8;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:25px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#f8fbff;
}

/* Text Area */
textarea{
    border-radius:12px !important;
    border:2px solid #3b82f6 !important;
}

/* Button */
div.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#2563eb,#1d4ed8);
    color:white;
    border:none;
    border-radius:12px;
    padding:14px;
    font-size:18px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:linear-gradient(90deg,#1d4ed8,#1e40af);
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Text Preprocessing --------------------
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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------- Load Model --------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# -------------------- Sidebar --------------------
with st.sidebar:

    st.title("📌 Project Details")

    st.markdown("""
### 🧠 Model
- Multinomial Naive Bayes

### 🔍 Feature Extraction
- TF-IDF Vectorizer

### 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- NLTK

---

Developed by **Sarthak Nigam**
""")

# -------------------- Header --------------------
st.markdown(
    '<div class="main-title">📩 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning powered Spam Classification using NLP</div>',
    unsafe_allow_html=True
)

# -------------------- Input --------------------
input_sms = st.text_area(
    "✉️ Enter your SMS Message",
    placeholder="Example: Congratulations! You have won ₹50,000. Click the link to claim..."
)

# -------------------- Prediction --------------------
if st.button("🚀 Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:

        with st.spinner("Analyzing message..."):

            transformed_sms = transform_text(input_sms)

            vector_input = tfidf.transform([transformed_sms])

            result = model.predict(vector_input)[0]

            probability = model.predict_proba(vector_input)

            confidence = probability.max() * 100

        st.markdown("---")

        if result == 1:

            st.error("🚨 Spam Message Detected")

        else:

            st.success("✅ This is a Ham Message")

        st.write("### Prediction Confidence")

        st.progress(int(confidence))

        st.write(f"**{confidence:.2f}% Confidence**")

# -------------------- Footer --------------------
st.markdown("---")

st.markdown(
    '<div class="footer">SMS Spam Detector | Built with Python, Streamlit & Scikit-learn</div>',
    unsafe_allow_html=True
)
