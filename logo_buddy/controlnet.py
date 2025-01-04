# !pip install opencv-python transformers accelerate
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import numpy as np
import torch

import cv2
from PIL import Image
from .utils import read_fit

MODEL_PATH = "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors"
CN_MODELS = {
    "canny": "/Users/drustsmith/repos/stable-diffusion-webui/models/ControlNet/control_canny-fp16.safetensors",
    "depth": "/Users/drustsmith/repos/stable-diffusion-webui/models/ControlNet/control_depth-fp16.safetensors",
    # "qr": "/Users/drustsmith/repos/stable-diffusion-webui/models/ControlNet/controlnetQRPatternQR_v2Sd15.safetensors",
}

CN_MODELS_XL = {
    "depth-xl": "depth_prepocessed",
    "canny-xl": "canny_preprocessed",
    # "canny-xl": "/Users/drustsmith/repos/stable-diffusion-webui/models/ControlNet/controlnet-canny-sdxl-1.safetensors",
}


def preprocess(image, controlnet_path=None):
    if "canny" in controlnet_path:
        return canny_preprocess(image)
    if "depth" in controlnet_path:
        # invert image
        image = 255 - np.array(image)
        # save copy of image
        # cv2.imwrite("depth.png", image)
        return Image.fromarray(image)
    else:
        return Image.fromarray(image)


def canny_preprocess(image):
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


if __name__ == "__main__":
    for cn, cn_path in CN_MODELS.items():
        # download an image
        image = read_fit("./input/bikedog_simple.png")

        # get processed image
        processed_image = preprocess(image, controlnet_path=cn_path)

        # load control net and stable diffusion v1-5
        controlnet = ControlNetModel.from_single_file(
            cn_path, torch_dtype=torch.float16, use_safetensors=True, device="mps"
        )

        pipe = StableDiffusionControlNetPipeline.from_single_file(
            MODEL_PATH,
            use_safetensors=True,
            torch_dtype=torch.float16,
            controlnet=controlnet,
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed
        # pipe.enable_xformers_memory_efficient_attention()

        # pipe.enable_model_cpu_offload()

        # generate image
        generator = torch.manual_seed(0)
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()

        image = pipe(
            "paper cut, paper layers, laser cut, paper art, vibrant colors, ",
            num_inference_steps=26,
            generator=generator,
            image=processed_image,
        ).images[0]
        image.save(f"controlnetout_{cn}.png")
