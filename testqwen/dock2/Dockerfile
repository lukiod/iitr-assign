FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git
    

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    torchao \
    git+https://github.com/huggingface/transformers.git \
    diffusers \
    Pillow \
    byaldi \
    qwen_vl_utils \
    flash-attn \
    fastapi \
    "uvicorn[standard]"

# Install additional required libraries
RUN pip install byaldi qwen-vl-utils

# Copy your application code
COPY app.py .


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]