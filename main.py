import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback
from transformers import DefaultDataCollator, create_optimizer

from metrics import SparseAUROC, SparseF1Score, SparsePrecision, SparseRecall
from utils import (
    create_image_folder_dataset,
    create_model,
    create_moka_dataset,
    create_preprocess,
    get_args,
)


def main():
    args = get_args()
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # or create dataset from imagefolder
    train_ds, val_ds = create_moka_dataset(
        args.image_path, [args.train_json, args.val_json]
    )
    train_ds = train_ds.shuffle(seed=42)

    preprocess = create_preprocess(args.model_id)

    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)

    data_collator = DefaultDataCollator(return_tensors="tf")

    tf_train_dataset = train_ds.to_tf_dataset(
        columns=["pixel_values"],
        label_cols=["label"],
        collate_fn=data_collator,
        batch_size=args.bs,
        shuffle=True,
    )
    tf_val_dataset = val_ds.to_tf_dataset(
        columns=["pixel_values"],
        label_cols=["label"],
        collate_fn=data_collator,
        batch_size=args.bs,
        shuffle=False,
    )

    num_train_steps = len(train_ds) // args.bs * args.num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=args.lr,
        num_train_steps=num_train_steps,
        weight_decay_rate=args.wd,
        num_warmup_steps=args.warmup_steps,
    )

    class_labels = train_ds.features["label"].names
    num_labels = len(class_labels)
    id2label = {i: label for i, label in enumerate(class_labels)}
    label2id = {label: i for i, label in enumerate(class_labels)}

    model = create_model(args.model_id, num_labels, id2label, label2id)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        SparsePrecision(num_classes=num_labels),
        SparseRecall(num_classes=num_labels),
        SparseF1Score(num_classes=num_labels),
        SparseAUROC(num_classes=num_labels),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    args.output_dir = os.path.join(args.output_dir, args.model_id)
    
    train_results = model.fit(
        tf_train_dataset,
        validation_data=tf_val_dataset,
        callbacks=[TensorboardCallback(log_dir=os.path.join(args.output_dir, "logs"))],
        epochs=args.num_train_epochs,
    )
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
