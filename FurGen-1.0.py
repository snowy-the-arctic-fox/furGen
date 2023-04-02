import os
os.environ['HF_HOME'] = 'D:\\cache'
import tensorflow as tf
import numpy as np
import PIL.Image
import json
from transformers import CLIPProcessor, AutoConfig
from transformers import TFAutoModel
from transformers import AutoTokenizer

# Load the model configuration from a URL
config_url = "https://huggingface.co/lunarfish/furrydiffusion/resolve/main/config.json"
config_path = tf.keras.utils.get_file("config.json", config_url, cache_dir="D:\\cache")
with open(config_path) as f:
    content = f.read()
config = AutoConfig.from_pretrained("lunarfish/furrydiffusion", from_dict=json.loads(content))

# Load the model weights from the Hugging Face model hub
generator = TFAutoModel.from_pretrained("lunarfish/furrydiffusion", config=config)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create the diffusion pipeline
print("creating diffusion pipeline")
tokenizer = AutoTokenizer.from_pretrained("lunarfish/furrydiffusion")

# Define function to generate image from text prompt
print("generating fursona")
def generate_fursona(prompt, truncation=0.5):
    # Generate text input
    input_text = tf.constant(prompt, tf.string)

    # Convert input text to tokens
    input_tokens = tokenizer(input_text, truncation=truncation, padding='max_length', max_length=128, return_tensors='tf')

    # Generate image from input tokens
    with tf.device('/cpu:0'):
        noise = tf.random.normal([1, 512])
        generated_image = generator.generate(
            input_ids=input_tokens['input_ids'],
            attention_mask=input_tokens['attention_mask'],
            random_noise=noise
        )

    # Postprocess image
    print("postprocessing")
    generated_image = (generated_image * 0.5 + 0.5) * 255
    generated_image = generated_image.numpy().astype(np.uint8)
    generated_image = PIL.Image.fromarray(generated_image[0])

    return generated_image

# Prompt user for text input
prompt = input("Fox with purple fur and green eyes: ")

# Generate and save image
print("Generating fursona image...")
image = generate_fursona(prompt)
image.save("D:\pfp\furGen(OUT1).png")
print("Image saved!")
