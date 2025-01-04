import os
from glob import glob
import random
import os.path as op
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
import numpy as np
import torch

import cv2

from .controlnet import preprocess, CN_MODELS, CN_MODELS_XL
from .utils import read_fit, OUT_DIR

STEPS = 34
SEED = -1
MODELS = {
    "real": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors",
    "anim": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/revAnimated_v122EOL.safetensors",
}
MODELS_XL = {
    "think-xl": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/thinkdiffusionxl_v10.safetensors",
}
LORAS = {
    "vintage_travel_poster": {
        "keywords": "vintage_travel_posters",
        "path": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/vintage_travel_posters_v1-000010.safetensors",
    },
    "bioshock_poster": {
        "keywords": "art deco, BioshockARTDECO",
        "path": "/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/BioshockARTDECO.safetensors",
    },
}

INPUT_DIR = "input/"

#
PROMPT_LIST = [
    {
        "text": "gardening, plants,  anime, studio ghibli",
        "loras": ["vintage_travel_poster"],
        "file_name": "plants",
    },
    {
        "text": """poster, modern""",
        "loras": ["bioshock_poster"],
        "file_name": "poster_bioshock",
    },
    {
        "text": """art, style, design, fancy""",
        "loras": ["bioshock_poster", "vintage_travel_poster"],
        "file_name": "art_both_loras",
    },
    {
        "text": "flowers, Random flowers, High Quality, Soft Texture, Full Colour, 80mm, ",
        "file_name": "flowers",
    },
    {"text": "flowers,  anime, studio ghibli, italy, spring", "file_name": "flowers"},
    {
        "text": "((ginger bread house)), candy, icing, realistic, insanely detailed, octane rendered, unreal engine, illustration, trending on artstation, masterpiece, photography",
        "file_name": "winter_ginger",
        "model": "real",
    },
    {
        "text": "((santa playing in the snow)), highly detailed, realistic lighting, sharp focus, rule of thirds, artgerm, wlop,  hd, octane, 4 k, ",
        "file_name": "winter_santa",
        "model": "anim",
    },  #  <lora:fantasy00d:0.5>, animated
    {
        "text": "ethereal fantasy concept art of dreamscape, surreal, ethereal, dreamy, mysterious, fantasy, highly detailed, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "file_name": "dreamscape",
    },
    {"text": "winter ice sculpture ", "file_name": "winter_ice"},
    # # General
    {
        "text": "a fun neon glowing sign",
        "loras": ["bioshock_poster", "vintage_travel_poster"],
        "file_name": "neon",
    },
    {"text": "hot air balloons ", "file_name": "hot_air_balloons", "model": "real"},
    {
        "text": "(wood carving), (inlay), (etsy) ",
        "file_name": "wood_carving",
        "model": "real",
    },
    {
        "text": "paper cut, paper layers, laser cut, paper art, vibrant colors, ",
        "file_name": "paper_art",
    },
    {
        "text": "carved halloween pumpkin, witches, spooky, fun, (vibrant colors:1.1), ",
        "file_name": "haloween",
        "model": "anim",
    },  #  <lora:fantasy00d:0.5>, animated
]

DEFAULT_POSITIVE_SUFFIX = ""
DEFAULT_NEGATIVE_PROMPT = "text, letters, nude, naked, porn, ugly, cleavage, extra hands, extra drawn feet, Extra fingers, poorly drawn face, fused fingers, long neck, big contrast, contrast white burn, white spots overexposed, over saturated, extra limbs, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, closed eyes, text, logo, fake 3D rendered image"


# env  is mac, cpu or gpu
DEVICE = "mps"
if torch.cuda.is_available():
    DEVICE = "cuda"


def get_pipe(model_path, controlnet_path=None, loras=None):
    controlnet_model = None
    if controlnet_path:
        if "xl" in model_path:
            if "canny" in controlnet_path:
                pretrained_path = "diffusers/controlnet-canny-sdxl-1.0"
            elif "depth" in controlnet_path:
                pretrained_path = "diffusers/controlnet-depth-sdxl-1.0"
            controlnet_model = ControlNetModel.from_pretrained(
                pretrained_path, torch_dtype=torch.float16
            )

            pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                model_path,
                use_safetensors=True,
                torch_dtype=torch.float16,
                controlnet=controlnet_model,
                safety_checker=None,
            )

        else:
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
                safety_checker=None,
            )
            if loras:
                for lora in loras:
                    # pipe.unet.load_attn_procs(LORAS[lora]["path"])
                    pipe.load_lora_weights(LORAS[lora]["path"])
                    pipe.fuse_lora(lora_scale=0.7)

    pipe = pipe.to(DEVICE)
    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    # Disable safety checker
    # def dummy(images, **kwargs):
    #     return images, False
    # pipe.safety_checker = dummy

    return pipe


def controlnet_generate(
    img_path, pipe, out_dir, prompts=PROMPT_LIST, controlnet=None, model_name=None
):
    if "xl" in model_name:
        image = read_fit(img_path, max_width=1024)
    else:
        image = read_fit(img_path)

    # add 10pixel margin on to cv2 image
    # Get color of top left pixel
    color = tuple(image[0, 0].astype(int).tolist())
    if "logo" in img_path.lower():
        BORDER_SIZE = 90
        image = cv2.copyMakeBorder(
            image,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=color,
        )
    elif image.shape[1] > (image.shape[0] * 1.5):
        # Close some of gap in ratio
        delta = image.shape[1] - image.shape[0]
        BORDER_SIZE = int(delta / 4)
        image = cv2.copyMakeBorder(
            image, BORDER_SIZE, BORDER_SIZE, 0, 0, cv2.BORDER_CONSTANT, value=color
        )
    # save cv2 image
    # cv2.imwrite(op.join(out_dir, f"original_{op.basename(img_path)}"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    preprocessed_image = None
    if controlnet:
        preprocessed_image = preprocess(image, controlnet_path=controlnet)

    # save copy of preprocessed_image PIL Image
    # preprocessed_image.save(op.join(out_dir, f"preprocessed_{op.basename(img_path)}"))

    for p in prompts:
        if SEED == -1:
            seed = random.randint(0, 9000)
        else:
            seed = SEED
        generator = torch.manual_seed(seed)
        for i in range(0, 1):
            print(p["text"])
            steps = STEPS
            image = pipe(
                p["text"] + DEFAULT_POSITIVE_SUFFIX,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=steps,
                generator=generator,
                image=preprocessed_image,
                guidance_scale=9 if "qr" in controlnet else 6.5,
                controlnet_conditioning_scale=1.4 if "qr" in controlnet else 0.9,
                strength=1,
                # cross_attention_kwargs={"scale": 0.6} #blend lora weight
            ).images[0]
            dest_path = op.join(
                out_dir, f"{p['file_name']}_{model_name}_{controlnet}_{seed}_{i}.png"
            )
            print("Save to ", dest_path)
            image.save(dest_path)


def make_logo(
    model_name, model_path, control_net_name, controlnet_path, lora_name=None
):
    print()
    pipe = get_pipe(model_path=model_path, controlnet_path=controlnet_path)

    input_imgs = (
        glob(op.join(INPUT_DIR, "**/*jpg"), recursive=True)
        + glob(op.join(INPUT_DIR, "**/*jpeg"), recursive=True)
        + glob(op.join(INPUT_DIR, "**/*png"), recursive=True)
    )

    # drop any in the skip folder
    input_imgs = [i for i in input_imgs if "skip" not in i.lower()]
    print("num imgs", len(input_imgs))

    if len(input_imgs) > 5:
        input_imgs = np.random.choice(input_imgs, 3, replace=False)

    for img_path in input_imgs:
        print(
            "model ",
            model_name,
            " | controlnet",
            control_net_name,
            " | ",
            op.basename(img_path),
        )
        out_dir = op.join(OUT_DIR, op.basename(img_path).split(".")[0])
        os.makedirs(out_dir, exist_ok=True)
        # out_dir = op.join(out_dir, op.basename(m).split(".")[0][:5])
        # os.makedirs(out_dir, exist_ok=True)

        # subset prompts to ones asking for this model, or no model specified.
        prompts = [p for p in PROMPT_LIST if p.get("model", m) == m]

        # This will be too many prompt combinations, so randomly down sample
        if len(prompts) > 5 and len(input_imgs) > 3:
            prompts = np.random.choice(prompts, 5, replace=False)

        for i, p in enumerate(prompts):
            if "loras" in p:
                for lora in p["loras"]:
                    prompts[i]["text"] += "," + LORAS[lora]["keywords"]

        print("prompts, ", prompts)
        controlnet_generate(
            img_path,
            pipe,
            controlnet=cn,
            out_dir=out_dir,
            prompts=prompts,
            model_name=m,
        )


# if main
if __name__ == "__main__":
    # run small models
    for m, mp in MODELS.items():
        print(
            "model:",
            m,
        )
        for cn, cn_path in CN_MODELS.items():
            make_logo(m, mp, cn, cn_path)

    # Run XL models.
    for m, mp in MODELS_XL.items():
        print(
            "model:",
            m,
        )
        for cn, cn_path in CN_MODELS_XL.items():
            make_logo(m, mp, cn, cn_path)
