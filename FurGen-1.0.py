import os
os.environ['HF_HOME'] = 'D:\\cache'
import tensorflow as tf
import numpy as np
import PIL.Image
import json
from transformers import CLIPProcessor, AutoTokenizer
import tensorflow.keras as keras

print("furGen-ON")

# Load the model configuration from a URL
print("getting request for config.json url")
config_url = "https://huggingface.co/lunarfish/furry-diffusion/resolve/main/config.json"
cache_dir = "D:\\cache"
config_path = keras.utils.get_file("config.json", config_url, cache_dir=cache_dir)
with open(config_path) as f:
    content = f.read()
print(content)
response = json.loads(keras.utils.get_file("config.json", config_url, cache_dir=cache_dir))
model_config = response

# Load the model weights from the Hugging Face model hub
print("loading model weights")
generator = AutoModelForConditionalGeneration.from_pretrained("lunarfish/furry-diffusion", config=model_config)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create the diffusion pipeline
print("creating diffusion pipeline")
tokenizer = AutoTokenizer.from_pretrained("lunarfish/furry-diffusion")

# Define function to generate image from text prompt
print("generating fursona")
def generate_fursona(prompt, truncation=0.5):
    # Generate text input
    print("generating text input:", prompt)
    input_text = tf.constant(prompt, tf.string)

    # Convert input text to tokens
    print("converting input text into tokens")
    input_tokens = tokenizer(input_text, truncation=truncation, padding='max_length', max_length=128, return_tensors='tf')

    # Generate image from input tokens
    print("generating image")
    with tf.device('/cpu:0'):
        generated_image = generator.generate(input_ids=input_tokens['input_ids'],
                                              attention_mask=input_tokens['attention_mask'],
                                              max_length=128,
                                              num_beams=1,
                                              no_repeat_ngram_size=2,
                                              early_stopping=True)[0]

    # Postprocess image
    print("postprocessing", generated_image)
    generated_image = generated_image.numpy()
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min()) * 255
    generated_image = generated_image.astype(np.uint8)
    generated_image = PIL.Image.fromarray(generated_image)

    return generated_image

# Prompt user for text input
prompt = input("Fox with purple fur and green eyes: ")

# Generate and save image
print("Generating fursona image...")
image = generate_fursona(prompt)
image.save("D:\\pfp\\furGen(OUT1).png")
print("Image saved!")
