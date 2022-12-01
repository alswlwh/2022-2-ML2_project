import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np

class Labeled_data(Dataset):
    
    def __init__(self, root, train=False, transform=None):
        super(Labeled_data, self).__init__()
        self.make_dataset(root, train)
        self.train = train
        self.transform = transform

    def make_dataset(self, root, train):
        self.data = []

        if train == True:

            root = f'{root}/train'

            categories = os.listdir(root)
            categories = sorted(categories)
        
        else:

            root = f'{root}/val'

            categories = os.listdir(root)
            categories = sorted(categories)

        for label, category in enumerate(categories):
            
            images = glob.glob(f'{root}/{category}/*.jpg')

            for image in images:

                self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = self.read_image(image)

        if self.transform is not None:

            image = self.transform(image)

        return image, label

    def read_image(self, path):
        HEIGHT = 64
        WIDTH = 64

        image = Image.open(path)
        image = image.resize((HEIGHT, WIDTH))

        return image.convert('RGB')


class Unlabeled_data(Dataset):
    
    def __init__(self, root, transform=None):
        super(Unlabeled_data, self).__init__()
        self.make_dataset(root)
        self.transform = transform

    def make_dataset(self, root):
        self.data = []

        categories = os.listdir(root)
        categories = sorted(categories)

        for label, category in enumerate(categories):
            
            images = glob.glob(f'{root}/{category}/*.jpg')

            for image in images:

                self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = self.read_image(image)

        if self.transform is not None:

            image = self.transform(image)

        return image

    def read_image(self, path):
        HEIGHT = 64
        WIDTH = 64

        image = Image.open(path)
        image = image.resize((HEIGHT, WIDTH))

        return image.convert('RGB')

class NS_labeled_data(Dataset):
    
    def __init__(self, root, train=True, transform=None):
        super(NS_labeled_data, self).__init__()
        self.make_dataset(root, train)
        self.train = train
        self.transform = transform

    def make_dataset(self, root, train):
        self.data = []

        if train == True:

            root = f'{root}/train'

            categories = os.listdir(root)
            categories = sorted(categories)
        
        else:

            root = f'{root}/val'

            categories = os.listdir(root)
            categories = sorted(categories)

        for label, category in enumerate(categories):
            
            images = glob.glob(f'{root}/{category}/*.jpg')

            for image in images:
                
                image = self.read_image(image)
                image = T.ToTensor()(image)
                self.data.append((image, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = T.ToPILImage()(image)

        if self.transform is not None:

            image = self.transform(image)

        return image, label

    def read_image(self, path):
        HEIGHT = 64
        WIDTH = 64

        image = Image.open(path)
        image = image.resize((HEIGHT, WIDTH))

        return image.convert('RGB')

class NS_pseudo_data(NS_labeled_data):
    
    def __init__(self, labeled_root, batch_image, model, device, transform=None):
        super(NS_pseudo_data, self).__init__(root=labeled_root, transform=transform)

   
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(batch_image.to(device))

        _, logits = torch.max(outputs, dim=1)

        for image, logit in zip(batch_image, logits):

            logit = logit.item()
            self.data.append((image, logit))
                 
            