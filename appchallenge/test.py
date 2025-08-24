from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Image captioning model
img_model_name = "nlpconnect/vit-gpt2-image-captioning"
img_model = VisionEncoderDecoderModel.from_pretrained(img_model_name).to(device)
img_feature_extractor = ViTImageProcessor.from_pretrained(img_model_name)
img_tokenizer = AutoTokenizer.from_pretrained(img_model_name)

# Step 2: Text expansion model
text_model_name = "google/flan-t5-base"
text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name).to(device)
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

def analyze_swimming(image_path):
    # --- Step 1: Get short caption ---
    image = Image.open(image_path).convert("RGB")
    pixel_values = img_feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = img_model.generate(pixel_values, max_length=20, num_beams=4)
    short_caption = img_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # --- Step 2: Expand into detailed coach feedback ---
    prompt = f"You are a professional swimming coach. Based on this description: '{short_caption}', give a detailed analysis including swimming style, body position, arm and leg movement, head position, breathing, equipment, and environment."
    inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
    expanded_ids = text_model.generate(inputs["input_ids"], max_length=200, num_beams=4)
    detailed_feedback = text_tokenizer.decode(expanded_ids[0], skip_special_tokens=True)

    return detailed_feedback
print(analyze_swimming("uploads/testimage.jpg"))
