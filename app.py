import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Text Classification", 
    page_icon=":guardsman:", 
    layout="wide"
)

st.title("Text Classification")

# Load models with better error handling
@st.cache_resource
def load_models():
    try:
        with open('models/pipeline_bow.pkl', 'rb') as file:
            sentiment_model = pickle.load(file)

        with open('models/sms_pipeline_bow.pkl', 'rb') as file:
            spam_model = pickle.load(file)
            
        return sentiment_model, spam_model
        
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

sentiment_model, spam_model = load_models()

# Sidebar navigation
option = st.sidebar.radio(
    "Select a Task:", 
    ("Sentiment Analysis", "Spam Detection", "About This Project"),
    key="task_selector"
)

if option == "Sentiment Analysis":
    st.title("Review Sentiment Analysis")
    review = st.text_area(
        "Enter Review Text:", 
        "Type your review here...", 
        height=200
    )
    
    if st.button("Predict Sentiment"):
        if review.strip() and review != "Type your review here...":
            with st.spinner("Analyzing sentiment..."):
                try:
                    prediction = sentiment_model.predict([review])
                    sentiment = "Positive 😄" if prediction[0] == 1 else "Negative 😞"
                    st.subheader(f"Prediction: {sentiment}")
                    
                    # Optional: Add confidence score if available
                    if hasattr(sentiment_model, 'predict_proba'):
                        proba = sentiment_model.predict_proba([review])[0]
                        st.write(f"Confidence: {max(proba)*100:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a review before clicking the 'Predict' button.")

elif option == "Spam Detection":
    st.title("SMS Spam Detection")
    message = st.text_area(
        "Enter SMS Text:", 
        "Type your message here...", 
        height=200, 
        key="sms_input"
    )
    
    if st.button("Predict Spam"):
        if message.strip() and message != "Type your message here...":
            with st.spinner("Checking for spam..."):
                try:
                    prediction = spam_model.predict([message])
                    classification = "Ham 📩" if prediction[0] == 1 else "Spam 🚫"
                    st.subheader(f"Prediction: {classification}")
                    
                    # Optional: Add confidence score if available
                    if hasattr(spam_model, 'predict_proba'):
                        proba = spam_model.predict_proba([message])[0]
                        st.write(f"Confidence: {max(proba)*100:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a message before clicking the 'Predict' button.")

elif option == "About This Project":
    st.title("About This Project")
    st.markdown("""
        ### Sentiment Analysis Model:
        This sentiment analysis model is built using different techniques:
        
        - **Bag of Words (BoW) with Multinomial Naive Bayes (MNB)**: This model achieved an accuracy of **85%** on the test dataset.
        - **TF-IDF (Term Frequency-Inverse Document Frequency)**: This model achieved an accuracy of **70%**.
        - **Average Word2Vec**: This model achieved an accuracy of **75%**.
        
        These models work by analyzing the text's content to classify whether the sentiment is **Positive** or **Negative**.

        ### Spam Detection Model:
        The spam detection model uses:
        
        - **Bag of Words (BoW) with Multinomial Naive Bayes (MNB)**: This model achieved an accuracy of **98%**.
        - **TF-IDF (Term Frequency-Inverse Document Frequency)**: This model achieved an accuracy of **97%**.
        - **Word2Vec**: This model achieved an accuracy of **94%**.

        This model classifies SMS messages as either **Spam** or **Ham** based on their content.

        **Note:** The performance of these models varies due to the different ways each technique captures the semantic meaning of the text.
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    <div style="text-align:center; margin-top:2rem;">
        <p style="font-size:1.2rem; color:#00bfff;">
            Developed by <a href="https://github.com/divyaraj-vihol" 
            style="color:#00bfff; text-decoration:none; font-weight:bold;">
            Divyaraj Vihol</a>
        </p>
        <p style="font-size:0.9rem; color:#777;">
            Text Classification App | 2025
        </p>
    </div>
    """, unsafe_allow_html=True)