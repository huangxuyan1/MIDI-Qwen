python -m src.test_wrapper \
  --hf_disk_path /fs/scratch/PAS3150/gigamidi_filtered_bars8_notes100 \
  --mmm_config /users/PAS3150/alvinh/music_infilling/configs/tokenizer/tokenizer_100k.json \
  --qwen_config_dir /users/PAS3150/alvinh/music_infilling/configs/models/Qwen3-0.6B-Base \
  --out_json posids_sample.json