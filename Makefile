MODEL_PATH=/Users/drustsmith/repos/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors
MODEL_NAME=RealisticVision/
test:
	pytest tests

format:
	black .


control:
	# create sym link from model_path to model
	# ln -s $(MODEL_PATH) model.safetensors
	export PYTORCH_ENABLE_MPS_FALLBACK=1;
	poetry run python -m logo_buddy.controlnet

run:
	export PYTORCH_ENABLE_MPS_FALLBACK=1;
	poetry run python -m logo_buddy.main

convert:
	if [ ! -f convert_original_stable_diffusion_to_diffusers.py ]; then \
		wget https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/scripts/convert_original_stable_diffusion_to_diffusers.py; \
	fi
	poetry run python convert_original_stable_diffusion_to_diffusers.py \
		--checkpoint_path $(MODEL_PATH) \
		--dump_path $(MODEL_NAME) \
		--from_safetensors

sticker:
	poetry run python -m logo_buddy.sticker