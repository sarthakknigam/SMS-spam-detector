# SMS-spam-detector
A machine learning-based SMS Spam Detection web application built with Python, Scikit-learn, NLTK, and Streamlit.

Features--

- Detects whether an SMS is Spam or Ham
- Text preprocessing using NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes classifier
- Interactive web interface built with Streamlit
- Fast and lightweight prediction

 Tech Stack-

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Pickle
APP LINK-https://sms-spam-detector-sfeav3frl7wxm28rfypfr7.streamlit.app/
Project Structure

```
sms-spam-detector/
│── app.py
│── model.pkl
│── vectorizer.pkl
│── spam.csv
│── requirements.txt
│── README.md
│── sms-spam-detection.ipynb

📊 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains thousands of labeled SMS messages categorized as:

- Spam
- Ham (Not Spam)

Machine Learning Pipeline-

1. Data Cleaning
2. Text Preprocessing
3. Tokenization
4. Stopword Removal
5. Stemming
6. TF-IDF Vectorization
7. Model Training using Multinomial Naive Bayes
8. Spam/Ham Prediction

📈 Model Used

- Multinomial Naive Bayes



