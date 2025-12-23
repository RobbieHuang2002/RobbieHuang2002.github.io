---
title: "Hello World!"
date: 2025-12-23 00:00:00
categories: [Deep Learning]
tags: [ai]
---
# Lesson #1 

So I've been going through the first lesson on FastAI's course on deep learning, and the first assignment that we got was to just try and experiment with the FastAI library or even running through the Kaggle notebook again. 

# What I Built

So I decided to experiment with the Kaggle notebook that was used in the lesson. I started messing around with different datasets I could find on the internet, and stumbled upon this one on Roboflow, https://universe.roboflow.com/georgebrown/kick-and-punch-object-detection

It's a dataset of roughly 4,000 images of MMA fights labeling the types of moves that are being thrown by the fightes (punch, kick, grapple, stand). I decided to experiment and download the dataset in Yolov8 format to run it through a resnet18 model. 

# Outcome

It was cool to grab some images from youtube and put it through the model. 


![Descriptive alt text](/assets/img/2025-12-23.png)


```python
from fastai.vision.all import *
from fastdownload import download_url
from pathlib import Path
import yaml

base_path = Path("/kaggle/working/Kick-and-punch-object-detection-12")


with open(base_path/"data.yaml") as f:
    data = yaml.safe_load(f)
names = data["names"]  # e.g. ['grappling', 'kick', 'punch', 'stand']


def splitter(items):
    train_idxs = [i for i,o in enumerate(items) if "/train/images/" in str(o)]
    valid_idxs = [i for i,o in enumerate(items) if "/valid/images/" in str(o)]
    return train_idxs, valid_idxs

def get_train_valid_images(path):
    return get_image_files(path/"train/images") + get_image_files(path/"valid/images")

def yolo_lbl_path(img_path):
    return Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")

def multilabel_from_yolo(img_path: Path):
    lblp = yolo_lbl_path(img_path)
    if not lblp.exists() or lblp.stat().st_size == 0:
        return []
    cls_ids = []
    for line in lblp.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            cls_ids.append(int(float(parts[0])))
    cls_ids = sorted(set(cls_ids))
    return [names[i] for i in cls_ids]

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(vocab=names)),  # <-- encoded removed
    get_items=get_train_valid_images,
    splitter=splitter,
    get_y=multilabel_from_yolo,
    item_tfms=Resize(224, method="squish"),
    batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
)

dls = dblock.dataloaders(base_path, bs=32)
dls.show_batch(max_n=6)

learn = vision_learner(
    dls,
    resnet18,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[partial(accuracy_multi, thresh=0.5)]
)
learn.fine_tune(3)
```







