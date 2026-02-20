import os
# Force TensorFlow to use the older Keras 2 backend for compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Movie Critic", 
    page_icon="🎬", 
    layout="centered"
)

# Injecting Custom CSS for a "Premium" look
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 15px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus { border-color: #ff4b4b; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #ff4b4b;
        color: white;
        font-weight: 600;
        padding: 0.75rem 0;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transform: translateY(-2px);
    }
    h1 { font-weight: 800; color: #1f2937; }
    .footer {
        text-align: center;
        padding-top: 3rem;
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CACHING MODELS & DATA 
# ==========================================
class SafeSimpleRNN(SimpleRNN):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

@st.cache_resource(show_spinner="Loading AI Core...")
def load_assets():
    word_idx = imdb.get_word_index()
    rev_word_idx = {value: key for key, value in word_idx.items()}
    model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': SafeSimpleRNN})
    return word_idx, rev_word_idx, model

word_index, reverse_word_index, model = load_assets()


# ==========================================
# 3. HELPER FUNCTIONS (UPDATED WITH REGEX)
# ==========================================
def preprocess_text(text):
    # Strip punctuation and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    words = clean_text.split()
    
    encoded_review = []
    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2) # Safely assign <UNK> token for unknown words
            
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# ==========================================
# 4. SIDEBAR & DEVELOPER INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3172/3172568.png", width=80) 
    st.markdown("**Developed by: Babar Ali**")
    st.markdown("---")
    st.markdown("### 📌 How it works")
    st.markdown("""
    1. Type or paste a movie review.
    2. Click **Analyze Sentiment**.
    3. The Recurrent Neural Network (RNN) will score the emotional tone of the text.
    """)


# ==========================================
# 5. APP UI & LAYOUT 
# ==========================================
st.title('🎬 AI Movie Critic')
st.markdown("### *Discover the hidden sentiment behind any movie review.*")
st.markdown("Drop your movie review below, and our deep learning model will analyze the emotional tone in seconds.")
st.divider()

user_input = st.text_area(
    'Enter your review:',
    height=150,
    placeholder="e.g., 'This movie was an absolute masterpiece! The cinematography was stunning...'",
    help="Type or paste a movie review in English."
)

if st.button('Analyze Sentiment 🚀'):
    if not user_input.strip():
        st.warning('⚠️ Please enter a movie review before analyzing.', icon="🚨")
    else:
        with st.spinner('Analyzing semantic patterns...'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            
            # UPDATED: True Certainty Math (0% to 100%)
            score = float(prediction[0][0])
            confidence = abs(score - 0.5) * 2 * 100
            
            is_positive = score > 0.5
            sentiment_text = 'Positive' if is_positive else 'Negative'
        
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if is_positive:
                st.success(f"**Sentiment:** {sentiment_text} 🍿", icon="✅")
            else:
                st.error(f"**Sentiment:** {sentiment_text} 🍅", icon="❌")
                
        with col2:
            st.metric(
                label="AI Confidence Score", 
                value=f"{confidence:.1f}%",
                delta="High Certainty" if confidence > 50 else "Uncertain",
                delta_color="normal" if confidence > 50 else "off"
            )

st.markdown("<div class='footer'>Designed & Developed by <b>Babar Ali</b></div>", unsafe_allow_html=True)
