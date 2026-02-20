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
# 1. PAGE CONFIGURATION & FYP UI THEME
# ==========================================
st.set_page_config(
    page_title="DeepSent: AI Movie Critic", 
    page_icon="🌌", 
    layout="centered"
)

# Premium 3D & Glassmorphism CSS styling
st.markdown("""
    <style>
    /* Dark Premium Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* 3D Glassmorphism Text Area */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        color: #ffffff !important;
        box-shadow: inset 3px 3px 8px rgba(0, 0, 0, 0.5), inset -3px -3px 8px rgba(255, 255, 255, 0.02);
        padding: 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border: 1px solid #00d2ff !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.3), inset 3px 3px 8px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* 3D Tactile Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 0;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4), inset 0 2px 2px rgba(255, 255, 255, 0.4);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 20px rgba(0, 210, 255, 0.4), inset 0 2px 2px rgba(255, 255, 255, 0.4);
        color: white;
    }
    .stButton>button:active {
        transform: translateY(3px);
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.4);
    }
    
    /* Typography adjustments for dark theme */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    p {
        color: #d1d5db !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding-top: 4rem;
        color: #9ca3af;
        font-size: 0.9rem;
        letter-spacing: 1px;
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

@st.cache_resource(show_spinner="Booting Deep Learning Core...")
def load_assets():
    word_idx = imdb.get_word_index()
    rev_word_idx = {value: key for key, value in word_idx.items()}
    model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': SafeSimpleRNN})
    return word_idx, rev_word_idx, model

word_index, reverse_word_index, model = load_assets()


# ==========================================
# 3. BULLETPROOF HELPER FUNCTIONS 
# ==========================================
def preprocess_text(text):
    # Strip punctuation and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    words = clean_text.split()
    
    # Cap the vocabulary size to match training data and prevent Out of Bounds errors
    VOCAB_SIZE = 10000 
    
    encoded_review = []
    for word in words:
        if word in word_index:
            word_id = word_index[word] + 3
            if word_id < VOCAB_SIZE:
                encoded_review.append(word_id)
            else:
                encoded_review.append(2) # Out of bounds -> assign <UNK>
        else:
            encoded_review.append(2) # Not in dictionary -> assign <UNK>
            
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# ==========================================
# 4. SIDEBAR & DEVELOPER INFO
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 DeepSent AI")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By:")
    st.markdown("**Babar Ali**")
    st.markdown("---")
    st.markdown("### ⚙️ System Specs")
    st.markdown("- **Architecture:** Recurrent Neural Network (RNN)")
    st.markdown("- **Dataset:** IMDB 50K Movie Reviews")
    st.markdown("- **Framework:** TensorFlow / Keras")
    st.markdown("- **Vocab Limit:** Top 10,000 words")


# ==========================================
# 5. APP UI & LAYOUT 
# ==========================================
st.title('🌌 DeepSent: Neural Sentiment Analysis')
st.markdown("### *Uncovering the human emotion behind the text.*")
st.markdown("Powered by a custom Recurrent Neural Network. Enter a movie review below to process its semantic sequence.")
st.write("") # Spacer

user_input = st.text_area(
    'Input Text Sequence:',
    height=150,
    placeholder="e.g., 'The cinematography was absolutely breathtaking, but the plot fell flat...'",
)

if st.button('Initialize Analysis ⚡'):
    if not user_input.strip():
        st.warning('⚠️ Awaiting text input for analysis.', icon="🚨")
    else:
        with st.spinner('Running sequence through neural layers...'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            
            # True Certainty Math (0% to 100%)
            score = float(prediction[0][0])
            confidence = abs(score - 0.5) * 2 * 100
            
            is_positive = score > 0.5
            sentiment_text = 'Positive' if is_positive else 'Negative'
        
        st.write("") # Spacer
        st.subheader("📊 Network Output")
        col1, col2 = st.columns(2)
        
        with col1:
            if is_positive:
                st.success(f"**Classification:** {sentiment_text} 🟢", icon="✅")
            else:
                st.error(f"**Classification:** {sentiment_text} 🔴", icon="❌")
                
        with col2:
            st.metric(
                label="Model Certainty", 
                value=f"{confidence:.1f}%",
                delta="High Confidence" if confidence > 50 else "Ambiguous Data",
                delta_color="normal" if confidence > 50 else "off"
            )

st.markdown("<div class='footer'> Developed By: <b>Babar Ali</b></div>", unsafe_allow_html=True)
