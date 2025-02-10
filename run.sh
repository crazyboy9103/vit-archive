#!/bin/bash

source .venv/bin/activate

patch_size=$1
if [ "$patch_size" == "16" ]; then
  model_id="google/vit-base-patch16-224-in21k"
elif [ "$patch_size" == "32" ]; then
  model_id="google/vit-base-patch32-224-in21k"
else
  echo "Invalid patch size. Please provide either 16 or 32."
  exit 1
fi

base=/mnt/d/datasets/Medical

# results will be logged to $output_dir/$model_id/logs
python main.py --image_path "$base/images" \
               --train_json "$base/label_train.json" \
               --val_json "$base/label_test.json" \
               --output_dir /home/research/vit/output \
               --model_id "$model_id" \
               --bs 8 \
               --lr 0.0003 \
               --wd 0.01 \
               --warmup_steps 0 \
               --num_train_epochs 10 \
               --fp16