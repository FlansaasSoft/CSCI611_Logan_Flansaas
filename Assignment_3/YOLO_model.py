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


if __name__ == "__main__":
    model = YOLO("yolov8m.pt")