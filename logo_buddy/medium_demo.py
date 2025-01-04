from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import numpy as np
import torch

import cv2
from PIL import Image

DEVICE = "mps"
if torch.cuda.is_available():
    DEVICE = "cuda"

CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
# CONTROLNET_ID = "lllyasviel/sd-controlnet-depth"
MODEL_ID = "frankjoshua/realisticVisionV51_v51VAE"
RESOLUTION = 512


def canny_preprocess(image):
    """Prepocess image for canny controlnet."""
    # Run Canny
    image = cv2.Canny(image, 100, 200)

    # Reformat image to structure expected by pipeline
    image = image[:, :, None]
    # Make it 3 channel
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def read_fit(img_path, max_width=RESOLUTION):
    """Read image and resize to fit model."""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize image to X width, keep ratio
    h, w, _ = image.shape
    new_w = max_width
    new_h = int(h * (new_w / w))
    image = cv2.resize(image, (new_w, new_h))
    return image


if __name__ == "__main__":
    # Instantiate control net
    controlnet_model = ControlNetModel.from_pretrained(
        CONTROLNET_ID, use_safetensors=True, device=DEVICE, torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        # use_safetensors=True,
        torch_dtype=torch.float16,
        controlnet=controlnet_model,
    )

    # if on a macos device, use mps
    pipe = pipe.to(DEVICE)

    input_image = read_fit("input/hugging-logo.png")
    # Turn image in to canny image
    if "canny" in CONTROLNET_ID:
        processed_image = canny_preprocess(input_image)
        processed_image.save("canny_image.png")
    else:
        processed_image = Image.fromarray(input_image)
        # invert  image
        processed_image = processed_image.convert("L")
        processed_image = processed_image.point(lambda x: 255 - x)
        processed_image.save("depth_image.png")

    # Run the pipeline
    for i in range(0, 5):
        image = pipe(
            "logo of spring, flowers, garden, fantasy, sci-fi",
            negative_prompt="blurry, watermark",
            num_inference_steps=34,
            image=processed_image,
            guidance_scale=6.5,
            controlnet_conditioning_scale=1.15,
            strength=1,
        ).images[0]
        image.save(f"canny_{i}.png")
