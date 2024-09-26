# OCR and Document Search Web Application

This application performs Optical Character Recognition (OCR) on images containing Hindi and English text, and provides a keyword search functionality. It offers two OCR approaches:

## OCR Approaches

### 1. Qwen2 Model (GPU Required)

- **GPU Requirement**: CUDA-capable GPU (tested on NVIDIA RTX 3080)
- Features:
  - High accuracy OCR for Hindi and English text
  - Uses FlashAttention library
  - Ideal for complex or high-resolution images
- **Implementation**: 
  - Tested with Streamlit and Gradio interfaces
  - Available on Hugging Face Space: [Your Hugging Face Space Link]
- **Note**: This approach requires GPU access, which may limit deployment options on some platforms

### 2. General OCR Theory (GOT) Model (CPU Compatible)

- Standard CPU (no GPU required)
- Features:
  - OCR capability for Hindi and English text
  - More accessible for systems without GPUs
  - (Note: Currently in development)

## Features

- Image upload for OCR processing
- Extracted text retrieval
- Keyword search within extracted text
- Results display (varies based on interface: Streamlit, Gradio, or API)

## Technologies

- Python
- FastAPI (for API version)
- Streamlit and Gradio (for web interfaces)
- Hugging Face Transformers
- PyTorch
- Docker

## Setup and Installation

1. Clone the repository:
   ```
   git clone [Your Repository URL]
   cd ITR-ASSIGNMENT
   ```

2. Environment setup:
   - For Qwen2 (GPU required):
     ```
     cd testqwen/dock2
     docker build -t ocr-search-app .
     docker run --gpus all -p 8000:8000 ocr-search-app
     ```
   - For both approaches (local setup):
     ```
     cd testqwen/dock2
     pip install -r requirements.txt
     ```

## Usage

### Hugging Face Space
- Visit the Hugging Face Space link:
- Docker - https://huggingface.co/spaces/lukiod/dock2 
- Streamlit - https://huggingface.co/spaces/lukiod/streamlit_qwen
- Follow the interface instructions to upload images and perform OCR

### Local Streamlit or Gradio Interface
- Run the appropriate script for Streamlit or Gradio interface:
  ```
  streamlit run streamlit_app.py
  # or
  python gradio_app.py
  ```
- Access the web interface and follow on-screen instructions

### API Version (if applicable)
1. Start the FastAPI application:
   ```
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
2. Access API documentation: `http://localhost:8000/docs`
3. Use `/upload` endpoint to submit images for OCR
4. Use `/search` endpoint for keyword searches in extracted text

## Deployment Options

- Hugging Face Spaces: Already deployed for Approach 1
- GPU-Enabled Server (for Qwen2): e.g., AWS EC2 or Google Cloud with GPU or RTX 4090 or HX100 recommended in flash attention  
- Standard Server (for GOT): Any CPU-based cloud or VPS service
- Local Deployment: Suitable GPU for Qwen2, standard CPU for GOT

## Development

- `basic_qwen.ipynb`: For testing Qwen2 OCR (requires GPU)
- GOT approach: Under development, will be CPU-compatible



## Contact

[Mohak Gupta] - [mohakgupta0981@gmail.com]

Project Link: [Your Repository URL]
Hugging Face Space: [Your Hugging Face Space Link]