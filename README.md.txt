# Text Classification App

A Streamlit web application for performing sentiment analysis on reviews and spam detection on SMS messages.

## Features

- **Sentiment Analysis**: Classifies text reviews as Positive or Negative
- **Spam Detection**: Identifies whether an SMS message is Spam or Ham (legitimate)
- **Model Information**: Provides details about the different models and their performance

## Models Used

### Sentiment Analysis
1. **Bag of Words (BoW) with Multinomial Naive Bayes (MNB)**
   - Accuracy: 85%
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Accuracy: 70%
3. **Average Word2Vec**
   - Accuracy: 75%

### Spam Detection
1. **Bag of Words (BoW) with Multinomial Naive Bayes (MNB)**
   - Accuracy: 98%
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Accuracy: 97%
3. **Word2Vec**
   - Accuracy: 94%