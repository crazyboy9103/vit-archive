import argparse
import os

import datasets
from pycocotools.coco import COCO
from transformers import TFViTForImageClassification, ViTImageProcessor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    return parser.parse_args()


def create_model(model_id, num_labels, id2label, label2id):
    return TFViTForImageClassification.from_pretrained(
        model_id, num_labels=num_labels, id2label=id2label, label2id=label2id
    )


def create_preprocess(model_id):
    processor = ViTImageProcessor.from_pretrained(model_id)

    def preprocess(example):
        inputs = processor(example["img"], return_tensors="tf")
        inputs["label"] = example["label"]
        return inputs

    return preprocess


def create_moka_dataset(image_path, annotation_paths):
    dss = []
    for annotation_path in annotation_paths:
        coco = COCO(annotation_path)
        images = coco.loadImgs(coco.getImgIds())
        img_data_files = []
        label_data_files = []
        for image in images:
            filepath = os.path.join(image_path, image["file_name"])
            image_id = image["id"]
            try:
                ann_ids = coco.getAnnIds(imgIds=image_id)

                # only one annotation per image
                ann = coco.loadAnns(ann_ids)[0]

            except IndexError:
                print(f"No annotation found for image {filepath}")
                continue
            # get category id
            category_id = ann["category_id"]
            # get category name
            category_name = coco.loadCats(category_id)[0]["name"]

            img_data_files.append(filepath)
            label_data_files.append(category_name)

        categories = coco.cats
        _CLASS_NAMES = [categories[i]["name"] for i in range(len(categories))]
        features = datasets.Features(
            {
                "img": datasets.Image(),
                "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
            }
        )
        ds = datasets.Dataset.from_dict(
            {"img": img_data_files, "label": label_data_files}, features=features
        )
        dss.append(ds)
    return dss


def create_image_folder_dataset(root_path):
    """creates `Dataset` from image folder structure"""

    # get class names by folders names
    _CLASS_NAMES = os.listdir(root_path)
    # defines `datasets` features`
    features = datasets.Features(
        {
            "img": datasets.Image(),
            "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
        }
    )
    # temp list holding datapoints for creation
    img_data_files = []
    label_data_files = []
    # load images into list for creation
    for img_class in os.listdir(root_path):
        for img in os.listdir(os.path.join(root_path, img_class)):
            path_ = os.path.join(root_path, img_class, img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    # create dataset
    ds = datasets.Dataset.from_dict(
        {"img": img_data_files, "label": label_data_files}, features=features
    )
    return ds
