import torch
from numba import cuda
import time
from diffusers import StableDiffusionPipeline
#from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "modular architectural facade with windows in"
my_image = pipe(prompt).images[0]
my_image.save("output.png")


"""
torch.cuda.empty_cache()

depthPipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", torch_dtype=torch.float16,)
depthPipe.to("cuda")

depthImage = depthPipe(prompt="", image=my_image, negative_prompt="", strength=0.7).images[0]

my_image.save(f"{arch_style}{material}.png")
depthImage.save(f"{arch_style}{material}_depth.png")
"""
