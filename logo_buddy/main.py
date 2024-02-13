import os
from glob import glob
import os.path as op
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image
import torch

from .controlnet import preprocess, CN_MODELS
from .utils import read_fit, OUT_DIR

STEPS = 32
SEED = 50
MODELS = {
    "anim": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/revAnimated_v122EOL.safetensors",
    "real": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors",
}

INPUT_DIR = "input/piccino"

#
PROMPT_LIST = [
    # {"text": "gardening, plants,  anime, studio ghibli", "file_name": "plants"},
    # {"text": "mountain bike,  anime, studio ghibli", "file_name": "bike", "model":"anim"},
    # {"text": "mountain bike,  anime, studio ghibli, action shot, motion", "file_name": "bike_motion", "model":"anim"},

    {"text": "flowers,  anime, studio ghibli, italy, spring", "file_name": "flowers"},
    {"text": "italian country side", "file_name": "italy"},
    # {"text": "a kayak in a  pool,  anime, studio ghibli", "file_name": "kayak"},
    # {"text": "bright colors, colorful kayak in a  pool,  anime, studio ghibli", "file_name": "kayak", "model":"anim"},
    # # Winter
    # {"text": "((ginger bread house)), candy, icing, realistic, insanely detailed, octane rendered, unreal engine, illustration, trending on artstation, masterpiece, photography", "file_name": "winter_ginger", "model":"real"},
    # {"text": "((santa playing in the snow)), highly detailed, realistic lighting, sharp focus, rule of thirds, artgerm, wlop,  hd, octane, 4 k, ", "file_name": "winter_santa", "model":"anim"}, #  <lora:fantasy00d:0.5>, animated    
    {
        "text": "ethereal fantasy concept art of dreamscape, surreal, ethereal, dreamy, mysterious, fantasy, highly detailed, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "file_name": "dreamscape",
        "model": "anim",
    },
    # {"text": "winter ice sculpture ", "file_name": "winter_ice"},

    # # General
    {"text": "a fun neon glowing sign", "file_name": "neon"},
    # # {"text": "hot air balloons ", "file_name": "hot_air_balloons", "model":"real"},
    {"text": "(wood carving), (inlay), (etsy) ", "file_name": "wood_carving", "model":"real"},
    {
        "text": "easter, pastel, bunny, pintrest",
        "file_name": "easter",
        "model": "real",
    },
    {
        "text": "paper cut, paper layers, laser cut, paper art, vibrant colors, ",
        "file_name": "paper_art",
        "model": "real",
    },
    # # {"text": "carved halloween pumpkin, witches, spooky, fun, (vibrant colors:1.1), ", "file_name": "haloween", "model":"anim"}, #  <lora:fantasy00d:0.5>, animated
    # {
    #     "text": "fun textures and colours , logo, pixar,  orange and pink clouds blue sky, sun, happy vibes, subtle  lense flare, birds, lots of details, sticker,  ",
    #     "file_name": "clouds",
    #     "model": "anim",
    # },
    #     {
    #     "text": "dolphin, ocean, ((lisa frank)), illustrated, graphic novel, colorful, sunset, happy vibes, wallpaper, hd, 4 k,   ",
    #     "file_name": "dolphin",
    #     "model": "anim",
    # },
    #     {
    #     "text": "((neutrophils)), biology, science, cell, cytology, hematology, white blood cell, microscopy, textbook  ",
    #     "file_name": "neutrophils",
    # },
]

DEFAULT_POSITIVE_SUFFIX = (
    ",detailed, intricate, best quality, (highest quality, award winning:1.3),"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, low res, low resolution, watermark, person, face, girl"
)



# env  is mac, cpu or gpu
DEVICE = "mps"
if torch.cuda.is_available():
    DEVICE = "gpu"


def get_pipe(model_path, controlnet_path=None):
    controlnet_model = None
    if controlnet_path:
        # load control net and stable diffusion v1-5
        controlnet_model = ControlNetModel.from_single_file(
            controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device=DEVICE,
        )

    pipe = StableDiffusionControlNetPipeline.from_single_file(
        model_path,
        use_safetensors=True,
        torch_dtype=torch.float16,
        controlnet=controlnet_model,
          safety_checker = None,
    )

    pipe = pipe.to(DEVICE)
    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    # Disable safety checker
    # def dummy(images, **kwargs):
    #     return images, False
    # pipe.safety_checker = dummy

    return pipe


def controlnet_generate(img_path, pipe, out_dir, prompts=PROMPT_LIST, controlnet=None, model_name=None):
    image = read_fit(img_path)
    preprocessed_image = None
    if controlnet:
        preprocessed_image = preprocess(image, controlnet_path=controlnet)

    for p in prompts:
        generator = torch.manual_seed(SEED)
        for i in range(0, 1):
            print(p["text"])
            steps = STEPS
            image = pipe(
                p["text"] + DEFAULT_POSITIVE_SUFFIX,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=steps,
                # generator=generator,
                image=preprocessed_image,
                guidance_scale=9 if 'qr' in controlnet else 7.5,
                controlnet_conditioning_scale=1.4 if 'qr' in controlnet else 1.0,
                strength=1.0,
            ).images[0]

            image.save(op.join(out_dir, f"{p['file_name']}_{model_name}_{controlnet}_{SEED}.png"))


# if main
if __name__ == "__main__":
    for m, mp in MODELS.items():
        for cn, cn_path in CN_MODELS.items():
            print()
            pipe = get_pipe(model_path=mp, controlnet_path=cn_path)

            input_imgs = (glob(op.join(INPUT_DIR, "**/*jpg"), recursive=True) + 
            glob(op.join(INPUT_DIR, "**/*jpeg"), recursive=True) + 
            glob(op.join(INPUT_DIR, "**/*png"), recursive=True))
            print('num imgs', len(input_imgs))

            for img_path in input_imgs:
                print('model ', m, ' | controlnet', cn, ' | ',op.basename(img_path))
                out_dir = op.join(OUT_DIR, op.basename(img_path).split(".")[0])
                os.makedirs(out_dir, exist_ok=True)
                # out_dir = op.join(out_dir, op.basename(m).split(".")[0][:5])
                # os.makedirs(out_dir, exist_ok=True)

                # subset prompts to ones asking for this model, or no model specified.
                prompts = [p for p in PROMPT_LIST if p.get("model", m) == m]

                # This will be too many prompt CN combinations, so randomly down sample
                if len(prompts) > 3 and len(CN_MODELS.keys()) > 1:
                    prompts = np.random.choice(prompts, 3, replace=False)

                controlnet_generate(
                    img_path, pipe, controlnet=cn, out_dir=out_dir, prompts=prompts, model_name=m
                )
