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

	random.seed(randomSeed)
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
	
def splitDataset(dataDir, imgKeys, outputPath, testSize=0.1, validSize=0.2, randomSeed=42):
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
			trainFile.write(f"{dataDir}/images/{key}.jpg\n")
		for key in validKeys:
			validFile.write(f"{dataDir}/images/{key}.jpg\n")
		for key in testKeys:
			testFile.write(f"{dataDir}/images/{key}.jpg\n")

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

def validateModel(model, yamlPath, outputPath, runName="yolo8vnBaselineTest", imgSize=640, conf=0.25, iou=0.5):
	results = model.val(data=yamlPath, split="test", imgsz=imgSize, single_cls=True, conf=conf, iou=iou, save_txt=True, save_conf=True, save_json=True, project=outputPath, name=runName, exist_ok=True)
	print(results.box.map)

def trainModel(model, yamlPath, outputPath, runName="yolo8vnBaselineTrain", epochs=30, batch_size=8, imgSize=640, dataFraction=1, trainTime=4, numWorkers=2, hsvH=0.015, hsvS=0.7, hsvV=0.4, degrees=0.0, translate=0.1, fliplr=0.5, scale=0.5, shear=0.0, perspective=0.0, mosaic=1.0, mixup=0.0, erase=0.4):
	model.train(data=yamlPath, epochs=epochs, time=trainTime, patience=5,batch=batch_size, imgsz=imgSize, fraction=dataFraction, single_cls=True, project=outputPath, name=runName, exist_ok=True, workers=numWorkers, hsv_h=hsvH, hsv_s=hsvS, hsv_v=hsvV, degrees=degrees, translate=translate, fliplr=fliplr, scale=scale, shear=shear, perspective=perspective, erasing=erase, mosaic=mosaic, mixup=mixup)

if __name__ == "__main__":
	PREPDATA = False  # Set to True to prepare data (convert annotations, split dataset, create YAML), False to skip and assume it's already done
	TRAIN = False  # Set to True to enable training, False to only run validation
	baselineModel = YOLO("yolov8n.pt")
	retrainedModel = YOLO("yolov8n.pt")
	highResModel = YOLO("yolov8n.pt")
	dataAugmentedModel = YOLO("yolov8n.pt")
	dataDir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

	gtSampleImg = visualize_gt("-0zd9UVk577mVT2hpqPEEQ", load_annotation("-0zd9UVk577mVT2hpqPEEQ", annotation_dir=dataDir + "/annotations"), image_dir=dataDir + "/images")
	plt.imshow(gtSampleImg)
	plt.axis('off')
	plt.show()
	sampleImgPath = dataDir + "/images/-0zd9UVk577mVT2hpqPEEQ.jpg"
	results = baselineModel(sampleImgPath)
	annotatedImg = results[0].plot()
	plt.imshow(annotatedImg[...,::-1])
	plt.axis('off')
	plt.show()

	yoloAnnoDir = dataDir + "/labels"
	splitsDir = dataDir + "/splits"
	annotationDir, imgDir, annoDict, imgKeys = getDirsKeysAndAnnos(dataDir)
	yamlPath = createDataYAML(dataDir, dataDir)

	if PREPDATA == True:
		makeMTSDAnnotatedSamples(annoDict, imgDir, imgKeys, "outputs/gt_samples", numSamples=10)
		prepareYOLOAnnotations(annoDict, imgKeys, yoloAnnoDir)
		splitDataset(dataDir, imgKeys, splitsDir)
	if TRAIN == True:
		trainModel(retrainedModel, yamlPath, "outputs", runName="yolo8vnRetrained", imgSize=416, dataFraction=0.2, hsvH=0.0, hsvS=0.0, hsvV=0.0, degrees=0.0, translate=0.0, fliplr=0.0, scale=0.0, shear=0.0, perspective=0.0, mosaic=0, mixup=0.0, erase=0.0) # No data augmentation
		trainModel(dataAugmentedModel, yamlPath, "outputs", runName="yolo8vnDataAugmented", imgSize=416, dataFraction=0.2, hsvH=0.015, hsvS=0.7, hsvV=0.4, degrees=10.0, translate=0.1, fliplr=0.5, scale=0.5, shear=0.0, perspective=0.0, mosaic=1.0, mixup=0.0, erase=0.4) # With data augmentation
		trainModel(highResModel, yamlPath, "outputs", runName="yolo8vnHighRes", imgSize=640, dataFraction=0.2, hsvH=0.015, hsvS=0.7, hsvV=0.4, degrees=10.0, translate=0.1, fliplr=0.5, scale=0.5, shear=0.0, perspective=0.0, mosaic=1.0, mixup=0.0, erase=0.4) # With data augmentation and higher resolution

	#validateModel(baselineModel, yamlPath, "outputs")

	retrainedModel = YOLO("runs/detect/outputs/yolo8vnRetrained/weights/best.pt")
	dataAugmentedModel = YOLO("runs/detect/outputs/yolo8vnDataAugmented/weights/best.pt")
	highResModel = YOLO("runs/detect/outputs/yolo8vnHighRes/weights/best.pt")

	confTestList = [0.25, 0.5, 0.75]
	iouTestList = [0.5, 0.65, 0.8]
	for conf in confTestList:
		for iou in iouTestList:
			print(f"Evaluating retrained model with conf={conf} and iou={iou}...")
			validateModel(retrainedModel, yamlPath, "outputs", runName=f"yolo8vnRetrainedTest_conf{conf}_iou{iou}", imgSize=416, conf=conf, iou=iou)
			print(f"Evaluating data augmented model with conf={conf} and iou={iou}...")
			validateModel(dataAugmentedModel, yamlPath, "outputs", runName=f"yolo8vnDataAugmentedTest_conf{conf}_iou{iou}", imgSize=416, conf=conf, iou=iou)
			print(f"Evaluating high res model with conf={conf} and iou={iou}...")
			validateModel(highResModel, yamlPath, "outputs", runName=f"yolo8vnHighResTest_conf{conf}_iou{iou}", imgSize=640, conf=conf, iou=iou)
	
	results = retrainedModel(sampleImgPath)
	annotatedImg = results[0].plot()
	plt.imshow(annotatedImg[...,::-1])
	plt.axis('off')
	plt.show()

	results = dataAugmentedModel(sampleImgPath)
	annotatedImg = results[0].plot()
	plt.imshow(annotatedImg[...,::-1])
	plt.axis('off')
	plt.show()

	results = highResModel(sampleImgPath)
	annotatedImg = results[0].plot()
	plt.imshow(annotatedImg[...,::-1])
	plt.axis('off')
	plt.show()