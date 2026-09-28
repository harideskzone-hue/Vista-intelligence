FROM python:3.11-slim

# Install system dependencies for OpenCV and AI models
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first for caching
COPY face_api/requirements.txt /app/face_api_requirements.txt
COPY face_engine/requirements.txt /app/face_engine_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/face_api_requirements.txt
RUN pip install --no-cache-dir -r /app/face_engine_requirements.txt
RUN pip install --no-cache-dir fast-plate-ocr ultralytics insightface onnxruntime faiss-cpu

# Copy the entire project
COPY . /app/

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# We need to run BOTH the FastAPI server and the Live Scorer (AI Engine)
# So we create a startup script and run it
RUN chmod +x /app/start_hf.sh

# Run the startup script
CMD ["/app/start_hf.sh"]
