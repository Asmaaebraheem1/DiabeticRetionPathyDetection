
# Diabetic Retinopathy Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/AsmaaElnagger/DiabeticRetionPathyDetection)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Asmaa111/diabetic-eye)


A Streamlit-based web application for detecting diabetic retinopathy from eye fundus images using a deep learning model from Hugging Face.


## 🔍 Overview
This application uses a pre-trained deep learning model to classify eye fundus images into different stages of diabetic retinopathy. The model is hosted on Hugging Face and integrated into a user-friendly Streamlit interface.

## 🚀 Getting Started

### Using Docker (Recommended)
1. Build the Docker image:
```bash
docker build -t diabetic-retinopathy .
```

2. Run the container:
```bash
docker run -p 8501:8501 diabetic-retinopathy
```

3. Access the app at `http://localhost:8501`

### Without Docker
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model:
```bash
python download_model.py
```

3. Run the Streamlit app:
```bash
streamlit run src/streamlit_app.py
```

## 🧠 Model Information
- **Model Name:** diabetic-eye
- **Repository:** [Asmaa111/diabetic-eye](https://huggingface.co/Asmaa111/diabetic-eye)
- **Framework:** PyTorch
- **Input:** Eye fundus images (JPEG/PNG)
- **Output:** Classification into retinopathy stages

## 📂 Project Structure
```
DiabeticRetionPathyDetection/
├── .streamlit/            # Streamlit configuration
├── src/
│   └── streamlit_app.py   # Main application code
├── Dockerfile             # Docker configuration
├── download_model.py      # Model download script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🌐 Live Demo
Try the live version hosted on Hugging Face Spaces:  
[DiabeticRetionPathyDetection Demo](https://huggingface.co/spaces/AsmaaElnagger/DiabeticRetionPathyDetection)

## 📚 Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Diabetic Retinopathy Detection Research](https://www.kaggle.com/c/diabetic-retinopathy-detection)
