import streamlit as st
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel
from PIL import Image
import io
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK data for BLEU score calculation
nltk.download('punkt', quiet=True)

# Load models and processors
@st.cache_resource
def load_models():
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    ).cuda().eval()
    
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    return RAG, qwen_model, qwen_processor

RAG, qwen_model, qwen_processor = load_models()

# Function to get current CUDA memory usage
def get_cuda_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2  # Convert to MB

# Define processing functions
def extract_text_with_colpali(image):
    start_time = time.time()
    start_memory = get_cuda_memory_usage()
    
    extracted_text = RAG.extract_text(image)
    
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

# Streamlit UI
st.title("Document Processing with ColPali and Qwen")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
query = st.text_input("Enter your query:")

if uploaded_file is not None and query:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process"):
        with st.spinner("Processing..."):
            # Extract text using ColPali
            colpali_extracted_text, colpali_metrics = extract_text_with_colpali(image)
            
            # Extract text using Qwen
            qwen_extracted_text, qwen_extract_metrics = process_with_qwen("", "", image, extract_mode=True)
            
            # Process the query with Qwen2, using both extracted text and image
            qwen_response, qwen_response_metrics = process_with_qwen(query, colpali_extracted_text, image)
            
            # Calculate BLEU score between ColPali and Qwen extractions
            bleu_score = calculate_bleu(colpali_extracted_text, qwen_extracted_text)

        # Display results
        st.subheader("Results")
        st.write("ColPali Extracted Text:")
        st.write(colpali_extracted_text)
        
        st.write("Qwen Extracted Text:")
        st.write(qwen_extracted_text)
        
        st.write("Qwen Response:")
        st.write(qwen_response)

        # Display metrics
        st.subheader("Metrics")
        
        st.write("ColPali Extraction:")
        st.write(f"Time: {colpali_metrics['time']:.2f} seconds")
        st.write(f"Memory: {colpali_metrics['memory']:.2f} MB")
        
        st.write("Qwen Extraction:")
        st.write(f"Time: {qwen_extract_metrics['time']:.2f} seconds")
        st.write(f"Memory: {qwen_extract_metrics['memory']:.2f} MB")
        
        st.write("Qwen Response:")
        st.write(f"Time: {qwen_response_metrics['time']:.2f} seconds")
        st.write(f"Memory: {qwen_response_metrics['memory']:.2f} MB")
        
        st.write(f"BLEU Score: {bleu_score:.4f}")

st.markdown("""
## How to Use

1. Upload an image containing text or a document.
2. Enter your query about the document.
3. Click 'Process' to see the results.

The app will display:
- Text extracted by ColPali
- Text extracted by Qwen
- Qwen's response to your query
- Performance metrics for each step
- BLEU score comparing ColPali and Qwen extractions
""")