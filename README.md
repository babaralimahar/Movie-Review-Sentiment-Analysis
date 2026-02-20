# 🎬 AI Movie Critic: Sentiment Analysis App

![App Screenshot](https://via.placeholder.com/800x400?text=Insert+Your+App+Screenshot+Here)

**[Live Demo: Click Here to Try the App!](#)** *(Update this link once deployed)*

## 📌 Overview
**AI Movie Critic** is a premium web application powered by Deep Learning that analyzes movie reviews and determines their emotional tone. Built with a Recurrent Neural Network (RNN) trained on the IMDB movie review dataset, this app provides real-time sentiment classification (Positive/Negative) along with an AI confidence score.

The user interface is designed with a modern, clean, and responsive aesthetic using Streamlit, featuring custom CSS, state caching for high performance, and an intuitive user experience.

---

## ✨ Features
* **Real-Time Analysis:** Instantly processes user input to predict sentiment.
* **Confidence Metric:** Displays exactly how confident the AI is in its prediction.
* **High Performance:** Utilizes Streamlit's `@st.cache_resource` to load the heavy Keras model into memory only once, resulting in lightning-fast predictions.
* **Premium UI/UX:** Custom-styled buttons, text areas, interactive spinners, and structured column layouts.
* **Error Handling:** Gracefully handles empty inputs and guides the user.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (RNN)
* **Web Framework:** Streamlit
* **Data Processing:** NumPy

---

## 🚀 Installation & Setup

Follow these steps to run the project on your local machine.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/ai-movie-critic.git](https://github.com/yourusername/ai-movie-critic.git)
cd ai-movie-critic
