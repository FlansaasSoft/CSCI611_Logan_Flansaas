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
import sys
import yaml

def load_annotation(image_key, annotation_dir='annotations'):
	with open(os.path.join(annotation_dir, '{:s}.json'.format(image_key)), 'r') as fid:
		anno = json.load(fid)
	return anno

def visualize_gt(image_key, anno, color='green', alpha=100, font=None, image_dir='images'):
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
			img_draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='red', width=2)

			class_name = obj['label']
			img_draw.text((x1 + 5, y1 + 5), class_name, font=font)

		img = Image.alpha_composite(img, rects)
		img = img.convert('RGB')

		return img

def getDirsKeysAndAnnos(dataPath):
	dataPath = Path(dataPath)
	annotationDir = next(p for p in dataPath.rglob("annotations") if p.is_dir())
	imgDir = next(p for p in dataPath.rglob("images") if p.is_dir())
	imgKeys = [f.stem for f in imgDir.glob("*.jpg")]
	annoDict = {key: load_annotation(key, annotation_dir=annotationDir) for key in imgKeys}
	return annotationDir, imgDir, annoDict, imgKeys

def makeMTSDAnnotatedSamples(annoDict, imgDir, imgKeys, outputPath, numSamples=10, randomSeed=42):
	outputPath = Path(outputPath)
	outputPath.mkdir(parents=True, exist_ok=True)

	selectedKeys = random.sample(imgKeys, min(numSamples, len(imgKeys)))

	for key in selectedKeys:
		img = visualize_gt(key, annoDict[key], image_dir=imgDir)
		img.save(outputPath / f"{key}.jpg")

def prepareYOLOAnnotations(annoDict, imgKeys, outputPath):
	outputPath = Path(outputPath)
	outputPath.mkdir(parents=True, exist_ok=True)

	for key in imgKeys:
		anno = annoDict[key]
		imgDimensions = (anno['width'], anno['height'])
		with open(outputPath / f"{key}.txt", 'w') as fid:
			for obj in anno['objects']:
				class_id = 0  # Assign single class 'traffic_sign'
				# Normalize coordinates to [0, 1] range
				x_center = (obj['bbox']['xmin'] + obj['bbox']['xmax']) / 2.0 / imgDimensions[0]
				y_center = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2.0 / imgDimensions[1]
				width = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / imgDimensions[0]
				height = (obj['bbox']['ymax'] - obj['bbox']['ymin']) / imgDimensions[1]
				fid.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
	
def splitDataset(imgKeys, outputPath, testSize=0.1, validSize=0.2, randomSeed=42):
	outputPath = Path(outputPath)
	outputPath.mkdir(parents=True, exist_ok=True)

	shuffleKeys = imgKeys.copy()
	random.seed(randomSeed)
	random.shuffle(shuffleKeys)

	splitIdx = int(len(shuffleKeys) * (1 - testSize - validSize))
	testSplitIdx = int(len(shuffleKeys) * (1 - testSize))
	trainKeys = shuffleKeys[:splitIdx]
	validKeys = shuffleKeys[splitIdx:testSplitIdx]
	testKeys = shuffleKeys[testSplitIdx:]

	with open(outputPath / "train.txt", 'w') as trainFile, open(outputPath / "val.txt", 'w') as validFile, open(outputPath / "test.txt", 'w') as testFile:
		for key in trainKeys:
			trainFile.write(f"images/{key}.jpg\n")
		for key in validKeys:
			validFile.write(f"images/{key}.jpg\n")
		for key in testKeys:
			testFile.write(f"images/{key}.jpg\n")

def createDataYAML(dataDir, outputPath):
	outputPath = Path(outputPath)
	outputPath.mkdir(parents=True, exist_ok=True)

	data_yaml = {
		'names': {0: 'traffic_sign'},
		'test': "splits/test.txt",
		'val': "splits/val.txt",
		'train': "splits/train.txt",
		'path': dataDir

	}
	with open(outputPath / "data.yaml", 'w') as yamlFile:
		yaml.dump(data_yaml, yamlFile)
	
	return outputPath / "data.yaml"

def testModel(model, yamlPath, outputPath, imgSize=640):
	results = model.val(data=yamlPath, split="test", imgsz=imgSize, single_cls=True, conf=0.25, iou=0.45, save_txt=True, save_conf=True, save_json=True, project=outputPath, name="yolo8vnBaselineTest", exist_ok=True)
	print(results)
		
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
	yoloAnnoDir = dataDir + "/labels"
	splitsDir = dataDir + "/splits"
	annotationDir, imgDir, annoDict, imgKeys = getDirsKeysAndAnnos(dataDir)
	makeMTSDAnnotatedSamples(annoDict, imgDir, imgKeys, "outputs/gt_samples", numSamples=10)
	prepareYOLOAnnotations(annoDict, imgKeys, yoloAnnoDir)
	splitDataset(imgKeys, splitsDir)
	yamlPath = createDataYAML(dataDir, dataDir)