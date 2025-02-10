source .venv/bin/activate

base=/mnt/d/datasets/Medical
python main.py --image_path $base/images \
               --train_json $base/label_train.json \
               --val_json $base/label_test.json \
               --output_dir /home/research/vit/output \
               --model_id google/vit-base-patch16-224-in21k \
               --bs 8 \
               --lr 0.00003 \
               --wd 0.01 \
               --warmup_steps 0 \
               --num_train_epochs 10 \
               --fp16