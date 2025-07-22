import streamlit as st
import numpy as np
import joblib
import cv2
import base64
import time
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load Model & Label Map 
model = joblib.load('best_model.pkl')
label_map = joblib.load('label_map.pkl')

# Feature Extraction Function 
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)

# Custom CSS Styling 
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    }
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #003366;
        margin-top: -40px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #333;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        margin-top: 40px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section 
st.markdown("<h1 class='main-title'>🛰️ Satellite Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a satellite image and get an instant land type prediction</p>", unsafe_allow_html=True)

# side bar
with st.sidebar:
    st.header("🔧 Settings")
    st.markdown("Choose your preferences below.")
    show_meta = st.checkbox("Show image metadata", value=True)
    show_model_info = st.checkbox("Show model information", value=False)

    st.markdown("---")
    st.markdown("📊 **Model:** SVM")
    st.markdown("🎯 **Features:** HSV Color Histogram (512 bins)")
    st.markdown("🧠 Trained on satellite images")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by Abhay Mittal")

# file upload
uploaded_file = st.file_uploader("📤 Upload a satellite image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="🖼️ Uploaded Image", use_column_width=True)

        if show_meta:
            st.markdown(f"📏 **Dimensions**: {pil_image.size[0]} x {pil_image.size[1]} px")
            st.markdown(f"💾 **File size**: {round(len(file_bytes) / 1024, 2)} KB")

        # Predict with Spinner 
        with st.spinner("🔍 Analyzing image..."):
            time.sleep(1.5)
            features = extract_features(image)
            prediction = model.predict(features)[0]
            label = label_map.inverse_transform([prediction])[0]

        # Output Block 
        st.markdown(f"""
        <div style='
            padding: 1.5rem;
            background-color: #e3f2fd;
            border-left: 8px solid #1e88e5;
            border-radius: 10px;
            margin-top: 20px;
        '>
            <h3 style='color: #0d47a1;'>📢 Predicted Class: <strong>{label}</strong></h3>
        </div>
        """, unsafe_allow_html=True)

        if show_model_info:
            with st.expander("🧠 View Model Info"):
                st.markdown("**Model:** Support Vector Machine (SVM)")
                st.markdown("**Features Used:** HSV Histogram (512 bins)")
                st.markdown("**Trained On:** Satellite Image Classification Dataset (Kaggle)")
                st.markdown("**Preprocessing:** Resized to 128x128, HSV histogram extraction")

    else:
        st.error("❌ Couldn't read the image. Please try again.")

else:
    st.info("👈 Upload a satellite image from your system.")

# --- Footer ---
st.markdown("""
<div class='footer'>
    🔗 Connect with me on 
    <a href='https://github.com/Abhay809' target='_blank'>GitHub</a> | 
    <a href='https://www.linkedin.com/in/abhay-kumar-mittal-9989b5255/' target='_blank'>LinkedIn</a><br>
    Made with using Streamlit
</div>
""", unsafe_allow_html=True)
