# MIDI-Qwen: Text-controlled Symbolic Music Infilling with Fill-in-middle Training

qwen3-pretrain/
├── configs/
│   ├── model/
│   └── train/
├── data/
│   ├── raw/
│   │   ├── gigamidi/
│   │   └── bach_doodle/
│   ├── processed/
│   └── meta/
├── scripts/
│   ├── preprocess/
│   ├── train/
│   │   └── train.sh
│   └── eval/
│       └── ppl_eval.py
├── src/
│   ├── train/
│   │   └── train_qwen_base_pe.py
│   ├── dataloader/
│   └── utils/
├── logs/
│   ├── tensorboard/
│   └── train_logs/
└── README.md
