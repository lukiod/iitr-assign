import streamlit as st
import torch
from PIL import Image
import gc
import tempfile
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel

# Function to load Byaldi model
@st.cache_resource
def load_byaldi_model():
    model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="cpu")
    return model

# Function to load Qwen2-VL model
@st.cache_resource
def load_qwen_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float32, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

# Function to clear GPU memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Streamlit Interface
st.title("OCR and Visual Language Model Demo")
st.write("Upload an image for OCR extraction and then ask a question about the image.")

# Image uploader
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # OCR Extraction with Byaldi
    st.write("Extracting text from image...")
    byaldi_model = load_byaldi_model()
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name

    # Create a temporary index for the uploaded image
    with st.spinner("Processing image..."):
        byaldi_model.index(temp_file_path, index_name="temp_index", overwrite=True)
    
    # Perform a dummy search to get the OCR results
    ocr_results = byaldi_model.search("Extract all text from the image", k=1)
    
    # Extract the OCR text from the results
    if ocr_results:
        extracted_text = ocr_results[0].metadata.get("ocr_text", "No text extracted")
    else:
        extracted_text = "No text extracted"
    
    st.write("Extracted Text:")
    st.write(extracted_text)
    
    # Clear Byaldi model from memory
    del byaldi_model
    clear_memory()

    # Remove the temporary file
    os.unlink(temp_file_path)

    # Text input field for question
    question = st.text_input("Ask a question about the image and extracted text")

    if question:
        st.write("Processing with Qwen2-VL...")
        qwen_model, qwen_processor = load_qwen_model()

        # Prepare inputs for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Extracted text: {extracted_text}\n\nQuestion: {question}"},
                ],
            }
        ]

        # Prepare for inference
        text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = qwen_processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")

        # Move tensors to CPU
        inputs = inputs.to("cpu")

        # Run the model and generate output
        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)

        # Decode the output text
        generated_text = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Display the response
        st.write("Model's response:", generated_text)

        # Clear Qwen model from memory
        del qwen_model, qwen_processor
        clear_memory()