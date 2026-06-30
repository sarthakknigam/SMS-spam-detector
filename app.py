import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- Download NLTK Data --------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#eef5ff,#dbeafe);
}

/* Main Title */
.main-title{
    text-align:center;
    font-size:44px;
    font-weight:700;
    color:#1e3a8a;
    margin-bottom:5px;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#4b5563;
    font-size:18px;
    margin-bottom:30px;
}

/* Text Area */
textarea{
    border-radius:12px !important;
    border:2px solid #2563eb !important;
}

/* Button */
div.stButton > button{
    width:100%;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
}

div.stButton > button:hover{
    background:#1d4ed8;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#f8fbff;
}

/* Footer */
.footer{
    text-align:center;
    color:#6b7280;
    font-size:14px;
    margin-top:30px;
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
### 🧠 Machine Learning Model
- Multinomial Naive Bayes

### 🔍 NLP Technique
- TF-IDF Vectorization

### 💻 Tech Stack
- Python
- Streamlit
- NLTK
- Scikit-learn

---

Developed by

**Sarthak Nigam**
""")

# -------------------- Header --------------------
st.markdown(
    '<div class="main-title">📩 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect Spam & Ham Messages using Machine Learning and Natural Language Processing</div>',
    unsafe_allow_html=True
)

# -------------------- Input --------------------
input_sms = st.text_area(
    "✉️ Enter your SMS Message",
    placeholder="Type or paste your SMS here..."
)

# -------------------- Prediction --------------------
if st.button("🚀 Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:

        with st.spinner("Analyzing message..."):

            transformed_sms = transform_text(input_sms)

            vector_input = tfidf.transform([transformed_sms])

            result = model.predict(vector_input)[0]

        st.markdown("---")

        if result == 1:
            st.error("🚨 Spam Message Detected")
            st.write("This message appears to be **spam**. Be cautious before responding or clicking any links.")
        else:
            st.success("✅ Ham Message")
            st.write("This message appears to be **legitimate** and does not exhibit common spam characteristics.")

# -------------------- Footer --------------------
st.markdown("---")

st.markdown(
    '<div class="footer">SMS Spam Detector | Built using Python, Streamlit, Scikit-learn & NLTK</div>',
    unsafe_allow_html=True
)
