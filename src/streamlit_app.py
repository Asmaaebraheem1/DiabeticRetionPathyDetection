import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
from io import BytesIO
from pdf2image import convert_from_bytes
import json
import os

# تأكد أن Streamlit تستخدم مجلد .streamlit القابل للكتابة
os.environ["STREAMLIT_HOME"] = "/app/.streamlit"

# Load Lottie animation
def load_lottiefile(filepath: str):
    if filepath.startswith(('http://', 'https://')):
        response = requests.get(filepath)
        response.raise_for_status()
        return response.json()
    else:
        with open(filepath, "r") as f:
            return json.load(f)

# Load the model and processor
# model_path = "Asmaa111/diabetic-eye" # new version 
model_path = "./diabetic_model" # for space
# model_path = r"C:\Users\Milestone\dinov2-base-finetuned-eye"
# model_path = r"AsmaaElnagger/Diabetic_RetinoPathy_detection" #  old version 


@st.cache_resource
def load_model():
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    return model, processor

def predict(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    predicted_label = model.config.id2label[predicted_class]
    return predicted_label

# App Config
st.set_page_config(
    page_title="Diabetic Eye Classifier",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS with animations
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section {
        padding: 4rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    .hero {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 5rem 1rem;
        border-radius: 10px;
        margin-bottom: 3rem;
    }
    .feature-card {
        padding: 2rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .team-card {
        text-align: center;
        padding: 1.5rem;
    }
    .testimonial-card {
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #4b6cb7;
    }
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .medical-info {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .image-section-prefix img{
            height: 300px ! IMPORTANT;
            max-width: 300px ! IMPORTANT;
            text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section with animation
with st.container():
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("👁️ Advanced Diabetic Retinopathy Screening")
        st.markdown("""
        **Revolutionizing** early detection of diabetic eye disease with **AI-powered** analysis.
        Get instant preliminary screening results from retinal images.
        """)
       
    st.markdown('</div>', unsafe_allow_html=True)

# Medical Information Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("ℹ️ About Diabetic Retinopathy", anchor="about-dr")
    
    with st.expander("What is Diabetic Retinopathy?"):
        st.markdown("""
        Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels 
        of the light-sensitive tissue at the back of the eye (retina). At first, diabetic retinopathy might cause no 
        symptoms or only mild vision problems. Eventually, it can cause blindness.
        """)
    
    with st.expander("Risk Factors"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            - **Duration of diabetes**: The longer you have diabetes, the greater your risk
            - **Poor blood sugar control**
            - **High blood pressure**
            - **High cholesterol**
            """)
        with cols[1]:
            st.markdown("""
            - **Pregnancy**
            - **Tobacco use**
            - **Being African-American, Hispanic or Native American**
            """)
    
    with st.expander("Prevention Tips"):
        st.markdown("""
        - **Manage your diabetes**: Keep your blood sugar levels in target range
        - **Monitor your blood sugar levels**
        - **Keep blood pressure and cholesterol under control**
        - **Quit smoking**
        - **Pay attention to vision changes**
        - **Have regular eye exams**
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# Features Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("✨ Key Features", anchor="features")
    cols = st.columns(3)
    features = [
        {"icon": "fa-brain", "title": "AI-Powered Analysis", 
         "desc": "Our deep learning model provides accurate preliminary screening in seconds"},
        {"icon": "fa-file-pdf", "title": "PDF Report Processing", 
         "desc": "Upload medical reports and we'll extract and analyze the images"},
        {"icon": "fa-shield-alt", "title": "Secure & Private", 
         "desc": "HIPAA compliant processing with no data retention"}
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="icon"><i class="fas {features[i]['icon']}"></i></div>
                <h3>{features[i]['title']}</h3>
                <p>{features[i]['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# How It Works Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("🔍 How It Works", anchor="how-it-works")
    steps = [
        {"title": "1. Upload Image", "desc": "Provide a retinal image or PDF report"},
        {"title": "2. AI Analysis", "desc": "Our model processes the image in seconds"},
        {"title": "3. Get Results", "desc": "Receive preliminary screening results"}
    ]
    for step in steps:
        with st.expander(step["title"]):
            st.write(step["desc"])
    st.markdown('</div>', unsafe_allow_html=True)


# Classifier Section
model, processor = load_model()

with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("📷 Try Our Classifier", anchor="classifier")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        option = st.radio(
            "Select input method:",
            ("Upload Image/PDF", "Image URL"),
            index=0
        )

        image = None
        if option == "Upload Image/PDF":
            uploaded_file = st.file_uploader(
                "Choose file",
                type=["jpg", "png", "jpeg", "pdf"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(uploaded_file.read())
                    if images:
                        image = images[0]
                else:
                    image = Image.open(uploaded_file)
        else:
            url = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
            if url:
                try:
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                except:
                    st.error("Invalid URL or image")

    with col2:
        if image:
            st.image(image, width=600)
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    prediction = predict(image, model, processor)
                    st.success(f"**Prediction:** {prediction}")
                    #st.balloons()
        else:
            st.info("Please upload an image or enter a URL to begin analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# Medical Disclaimer
with st.container():
    st.markdown("""
    <div class="medical-info">
    <h4>Medical Disclaimer</h4>
    <p>This tool provides preliminary screening only and is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.markdown("""
    **Contact Us**  
    <i class="fas fa-envelope"></i> contact@eyeai.com  
    <i class="fas fa-phone"></i> (123) 456-7890
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown("""
    **Quick Links**  
    [About DR](#about-dr)  
    [Features](#features)  
    [How It Works](#how-it-works)
    """)
with cols[2]:
    st.markdown("""
    © 2025 Diabetic Eye Classifier  
    Medical AI Application
    """)
