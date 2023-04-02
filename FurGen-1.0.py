import os
os.environ['HF_HOME'] = 'D:\\cache'
import tensorflow as tf
import numpy as np
import PIL.Image
import json
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPConfiguration, CLIPImageProcessor
from diffusers import StableDiffusionPipeline

# Load the model configuration
config = CLIPConfiguration.from_pretrained("openai/clip-vit-base-patch32")

# Create the CLIPImageProcessor
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create the diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "lunarfish/furrydiffusion", 
    config=config, 
    image_processor=image_processor,
    use_auth_token=True,
    token="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",
)

# Check if the model is already downloaded
if not os.path.exists("lunarfish/furrydiffusion"):
    # Download the pre-trained model
    print("Downloading model...")
    pipeline = DiffusionPipeline.from_pretrained("lunarfish/furrydiffusion", use_auth_token=False, from_tf=True)
else:
    # Load the pre-trained model from the local directory
    print("Loading model...")
    pipeline = DiffusionPipeline.from_pretrained("lunarfish/furrydiffusion", use_auth_token=True)

tokenizer = pipeline.tokenizer
generator = pipeline.model

# Define function to generate image from text prompt
def generate_fursona(prompt, truncation=0.5):
    # Generate text input
    input_text = tf.constant(prompt, tf.string)
    
    # Convert input text to tokens
    input_tokens = tokenizer(input_text, truncation=truncation, padding='max_length', max_length=128, return_tensors='tf')
    
    # Generate image from input tokens
    with torch.no_grad():
        noise = torch.randn([1, 512])
        generated_image = generator([noise, input_tokens['input_ids']])
    
    # Postprocess image
    generated_image = (generated_image * 0.5 + 0.5) * 255
    generated_image = generated_image.numpy().astype(np.uint8)
    generated_image = PIL.Image.fromarray(generated_image[0])
    
    return generated_image

# Prompt user for text input
prompt = input("Fox with purple fur and green eyes: ")

# Generate and save image
print("Generating fursona image...")
image = generate_fursona(prompt)
image.save("D:\pfp\furGen(OUT1.png)")
print("Image saved!")
