import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageColor, ImageFont
import pi_heif
import sys

def load_annotation(image_key, annotation_dir='annotations'):
    with open(os.path.join(annotation_dir, '{:s}.json'.format(image_key)), 'r') as fid:
        anno = json.load(fid)
    return anno

def visualize_gt(image_key, anno, color='green', alpha=125, font=None, image_dir='images'):
    try:
        font = ImageFont.truetype('arial.ttf', 15)
    except:
        print('Falling back to default font...')
        font = ImageFont.load_default()
    
    with Image.open(os.path.join(image_dir, '{:s}.jpg'.format(image_key))) as img:
        img = img.convert('RGBA')
        img_draw = ImageDraw.Draw(img)

        rects = Image.new('RGBA', img.size)
        rects_draw = ImageDraw.Draw(rects)

        for obj in anno['objects']:
            x1 = obj['bbox']['xmin']
            y1 = obj['bbox']['ymin']
            x2 = obj['bbox']['xmax']
            y2 = obj['bbox']['ymax']

            color_tuple = ImageColor.getrgb(color)
            if len(color_tuple) == 3:
                color_tuple = color_tuple + (alpha,)
            else:
                color_tuple[-1] = alpha

            rects_draw.rectangle((x1+1, y1+1, x2-1, y2-1), fill=color_tuple)
            img_draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='black', width=1)

            class_name = obj['label']
            img_draw.text((x1 + 5, y1 + 5), class_name, font=font)

        img = Image.alpha_composite(img, rects)
        img = img.convert('RGB')

        return img

def makeMTSDAnnotatedSamples(dataPath, outputPath, numSamples=10):
    dataPath = Path(dataPath)
    outputPath = Path(outputPath)
    annotationDir = next(p for p in dataPath.rglob("annotations") if p.is_dir())
    imgDir = next(p for p in dataPath.rglob("images") if p.is_dir())
    outputPath.mkdir(parents=True, exist_ok=True)

    image_keys = [f.stem for f in imgDir.glob("*.jpg")]
    selected_keys = random.sample(image_keys, min(numSamples, len(image_keys)))

    for key in selected_keys:
        anno = load_annotation(key, annotation_dir=annotationDir)
        img = visualize_gt(key, anno, image_dir=imgDir)
        img.save(outputPath / f"{key}.jpg")

def dataLoader(batch_size=32, valid_size=0.2, shuffle=True, random_seed=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root='mtsd_fully_annotated_images_train0/', transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_size * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    dataDir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    makeMTSDAnnotatedSamples(dataDir, "outputs/gt_samples", numSamples=10)