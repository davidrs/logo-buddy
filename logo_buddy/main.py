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
from .utils import read_fit

MODELS = {
    "real": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors",
    "anim": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/revAnimated_v122EOL.safetensors",
}

#
PROMPT_LIST = [
    # {"text": "a neon glowing sign", "file_name": "neon"},
    # {"text": "winter ice sculpture ", "file_name": "winter"},
    # {"text": "hot air balloons ", "file_name": "hot_air_balloons", "model":"real"},
    # {"text": "(wood carving), (inlay), (etsy) ", "file_name": "wood_carving", "model":"real"},
    {
        "text": "paper cut, paper layers, laser cut, paper art, vibrant colors, ",
        "file_name": "paper_art",
        "model": "real",
    },
    # {"text": "carved halloween pumpkin, witches, spooky, fun, (vibrant colors:1.1), ", "file_name": "haloween", "model":"anim"}, #  <lora:fantasy00d:0.5>, animated
    {
        "text": "fun textures and colours , logo, pixar,  orange and pink clouds blue sky, sun, happy vibes, subtle  lense flare, birds ",
        "file_name": "clouds",
        "model": "anim",
    },
]

DEFAULT_POSITIVE_SUFFIX = (
    ",detailed, intricate, best quality, (highest quality, award winning:1.3)"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, low res, low resolution, watermark, logo"
)


OUT_DIR = "./out"
os.makedirs(OUT_DIR, exist_ok=True)

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
    )

    pipe = pipe.to(DEVICE)
    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    return pipe


def controlnet_generate(img_path, pipe, out_dir, prompts=PROMPT_LIST, controlnet=None):
    image = read_fit(img_path)
    preprocessed_image = None
    if controlnet:
        preprocessed_image = preprocess(image, controlnet_path=controlnet)

    for p in prompts:
        generator = torch.manual_seed(0)
        for i in range(0, 1):
            print(DEFAULT_POSITIVE_SUFFIX)
            print(p["text"])
            steps = 32
            image = pipe(
                p["text"] + DEFAULT_POSITIVE_SUFFIX,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=steps,
                generator=generator,
                image=preprocessed_image,
            ).images[0]

            image.save(op.join(out_dir, f"{p['file_name']}_{steps}.png"))


# if main
if __name__ == "__main__":
    for m, mp in MODELS.items():
        for cn, cn_path in CN_MODELS.items():
            pipe = get_pipe(model_path=mp, controlnet_path=cn_path)
            input_imgs = glob("input/*")

            for img_path in input_imgs:
                out_dir = op.join(OUT_DIR, op.basename(img_path).split(".")[0])
                os.makedirs(out_dir, exist_ok=True)
                out_dir = op.join(out_dir, op.basename(m).split(".")[0][:5] + f"_{cn}")
                os.makedirs(out_dir, exist_ok=True)

                # subset prompts to ones asking for this model, or no model specified.
                prompts = [p for p in PROMPT_LIST if p.get("model", m) == m]

                controlnet_generate(
                    img_path, pipe, controlnet="canny", out_dir=out_dir, prompts=prompts
                )
