from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")
config.save_pretrained("/users/PAS3150/alvinh/music_infilling/configs/models/Qwen3-0.6B-Base")
