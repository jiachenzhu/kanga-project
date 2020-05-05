import os
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        
        self.images_name = sorted(os.listdir(root))
        self.name_list = [x.split('.')[0] for x in self.images_name]
        self.images = [Image.open(f"{root}/{x}").convert('RGB') for x in self.images_name]

    def __getitem__(self, index):
        img = self.images[index]

        results = []
        for transform in self.transform:
            results.append(transform(img))
        results.append(self.name_list[index])

        return results

    def __len__(self):
        return len(self.images)