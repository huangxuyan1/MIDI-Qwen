#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS3150
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=midiqwen
#SBATCH --mail-user=huang.5197@osu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=10:00:00
#SBATCH --mem=256GB

set -euo pipefail

source /users/PAS3150/alvinh/music_infilling/midi/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

which python
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
nvidia-smi

srun torchrun --standalone --nproc_per_node=1 \
  -m src.train_launcher \
  --dataset_name Metacreation/GigaMIDI \
  --dataset_config v2.0.0 \
  --mmm_config /users/PAS3150/alvinh/music_infilling/configs/tokenizer/tokenizer_100k.json \
  --qwen_config_dir /users/PAS3150/alvinh/music_infilling/configs/models/Qwen3-0.6B-Base \
  --train_config /users/PAS3150/alvinh/music_infilling/configs/train/train_stage1_ablation.json \
  --output_dir /users/PAS3150/alvinh/music_infilling/outputs/vanilla \
  --wandb_project midi_qwen \
  --wandb_run_name qwen3_stage1_gigamidi_filtered_exp_vanilla \
  --train_split train \
  --eval_split validation
