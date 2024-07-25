def slugify(text):
   text = re.sub(r"[^\w\s]", "", text)
   text = re.sub(r"\s+", "-", text)
   return text

import pyscreenshot
import torch
from numba import cuda
import time
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from pathlib import Path
from PIL import Image
import torch
import re

# Take a screenshot
screenshot = pyscreenshot.grab(bbox=(10, 10, 500, 500))

# View the screenshot
#image.show()

# Save the screenshot
screenshot.save("screenshot.png")

# Use screenshot in pipeline
images = ["sketch.png"]
init_images = [Image.open(image).convert("RGB").resize((512,512)) for image in images]

# Set the diffusion model
model_id = "stabilityai/stable-diffusion-2"
#model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "runwayml/stable-diffusion-inpainting"

# Set prompt to guide image towards
prompts = ["architecture, render, raytracing, arcane, detail, futuristic"]
#prompts = ["window, Modern Facade, Front View, Architectural, frontal, pin up, texture, seamless"]

# Set prompt to guide image away from
negative_prompts = ["blurry"]

# Image2image pipeline

# Set to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use the Euler scheduler here instead of default
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Choose pipeline
#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
#pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)

#
DIR_NAME="./images/"
dirpath = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath.mkdir(parents=True, exist_ok=True)

#
strength = 0.5
steps = 100
scale =7.5
num_images_per_prompt = 1
seed = torch.randint(0, 1000000, (1,)).item()
generator = torch.Generator(device=device).manual_seed(seed)

output = pipe(prompts, negative_prompt=negative_prompts, image=init_images, strength=strength, num_inference_steps=steps,
              guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)

#output = pipe(prompts, negative_prompt=negative_prompts, image=init_images, mask_image=init_images, num_inference_steps=steps,
#              guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)

#
for idx, (image,prompt) in enumerate(zip(output.images, prompts*num_images_per_prompt)):
    image_name = f'{slugify(prompt)}-{idx}.png'
    image_path = dirpath / image_name
    image.save(image_path)
    image.show()