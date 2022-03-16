import os
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class SingleImageDataset(Dataset):

    def __init__(self, train_file, image_dir, val=False, onehot=False):
        self.onehot = onehot
        self.val = val
        self.image_dir = image_dir

        if not self.val:
            self.train_file = train_file
            self.items = self.get_items()
        else:
            self.images = self.get_image_list(self.image_dir)

    def get_image_list(self, image_dir):
        images = []
        for file in os.listdir(image_dir):
            if '.png' in file:
                images.append(file)
        return images
    
    def get_items(self):
        items = []
        with open(self.train_file, 'r') as file:
            for line in file:
                line = line.split(',')
                line = [element.strip() for element in line]
                items.append(line)
        return items
    
    def __len__(self):
        if not self.val: # training
            return len(self.items)
        else: # validation
            return len(self.images)
    
    def __getitem__(self, idx):
        # training - use self.items
        if not self.val:
            item = self.items[idx]

            filename = item[0]

            image = os.path.join(self.image_dir, item[0])
            image = np.array(Image.open(image))[:, :, :-1] # remove alpha channel
            transforms = T.Compose([T.ToPILImage(),
                                    T.Resize((256,256)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transforms(image)
            
            if self.onehot:
                percentage = torch.from_numpy(np.array(int(item[1])))
                percentage = nn.functional.one_hot(percentage, num_classes=100)
            else:
                percentage = float(item[1])

            patient_id = int(item[2])

            return (image, percentage, patient_id, filename)
        
        # validation - use self.images
        else:
            image = self.images[idx]
            image_name = image

            image = os.path.join(self.image_dir, image)
            image = np.array(Image.open(image))[:, :, :-1] # remove alpha channel
            transforms = T.Compose([T.ToPILImage(),
                                    T.Resize((256,256)),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transforms(image)
            return (image, image_name)

if __name__ == '__main__':
    train_file = '../data/Train.csv'
    image_dir = '../data/Train'

    dataset = SingleImageDataset(train_file=train_file, image_dir=image_dir, onehot=True)