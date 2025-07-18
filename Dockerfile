FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
COPY download_model.py /app
COPY .streamlit/ /app/.streamlit/  
COPY src/ /app/src/

RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 download_model.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
