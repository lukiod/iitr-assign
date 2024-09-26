import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel
from PIL import Image
import io
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK data for BLEU score calculation
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Access environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
RAG_MODEL = os.getenv("RAG_MODEL", "vidore/colpali")
QWN_MODEL = os.getenv("QWN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
QWN_PROCESSOR = os.getenv("QWN_PROCESSOR", "Qwen/Qwen2-VL-2B-Instruct")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Initialize FastAPI app
app = FastAPI()

# Load models and processors
RAG = RAGMultiModalModel.from_pretrained(RAG_MODEL, use_auth_token=HF_TOKEN)

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWN_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
).cuda().eval()

qwen_processor = AutoProcessor.from_pretrained(QWN_PROCESSOR, trust_remote_code=True, use_auth_token=HF_TOKEN)

# Define request model
class DocumentRequest(BaseModel):
    text_query: str

# Function to get current CUDA memory usage
def get_cuda_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2  # Convert to MB

# Define processing functions
def extract_text_with_colpali(image):
    start_time = time.time()
    start_memory = get_cuda_memory_usage()
    
    # Use ColPali (RAG) to extract text from the image
    extracted_text = RAG.extract_text(image)  # Assuming this method exists
    
    end_time = time.time()
    end_memory = get_cuda_memory_usage()
    
    return extracted_text, {
        'time': end_time - start_time,
        'memory': end_memory - start_memory
    }

def process_with_qwen(query, extracted_text, image, extract_mode=False):
    start_time = time.time()
    start_memory = get_cuda_memory_usage()
    
    if extract_mode:
        instruction = "Extract and list all text visible in this image, including both printed and handwritten text."
    else:
        instruction = f"Context: {extracted_text}\n\nQuery: {query}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "image",
                    "image": image,
                },
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    end_time = time.time()
    end_memory = get_cuda_memory_usage()
    
    return output_text[0], {
        'time': end_time - start_time,
        'memory': end_memory - start_memory
    }

# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([reference_tokens], hypothesis_tokens)

# Define API endpoint
@app.post("/process_document")
async def process_document(request: DocumentRequest, file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    # Check API key
    if x_api_key != HF_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Read and process the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Extract text using ColPali
    colpali_extracted_text, colpali_metrics = extract_text_with_colpali(image)
    
    # Extract text using Qwen
    qwen_extracted_text, qwen_extract_metrics = process_with_qwen("", "", image, extract_mode=True)
    
    # Process the query with Qwen2, using both extracted text and image
    qwen_response, qwen_response_metrics = process_with_qwen(request.text_query, colpali_extracted_text, image)
    
    # Calculate BLEU score between ColPali and Qwen extractions
    bleu_score = calculate_bleu(colpali_extracted_text, qwen_extracted_text)
    
    return {
        "colpali_extracted_text": colpali_extracted_text,
        "qwen_extracted_text": qwen_extracted_text,
        "qwen_response": qwen_response,
        "metrics": {
            "colpali_extraction": colpali_metrics,
            "qwen_extraction": qwen_extract_metrics,
            "qwen_response": qwen_response_metrics,
            "bleu_score": bleu_score
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)