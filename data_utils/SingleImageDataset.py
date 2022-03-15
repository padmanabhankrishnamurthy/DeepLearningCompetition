import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SingleImageDataset(Dataset):

    def __init__(self, train_file, image_dir):
        self.train_file = train_file
        self.image_dir = image_dir
        self.items = self.get_items()

    def get_items(self):
        items = []
        with open(self.train_file, 'r') as file:
            for line in file:
                line = line.split(',')
                line = [element.strip() for element in line]
                items.append(line)
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]

        filename = item[0]

        image = os.path.join(self.image_dir, item[0])
        image = np.array(Image.open(image))[:, :, :-1] # remove alpha channel
        transforms = T.Compose([T.ToPILImage(),
                                T.Resize((256,256)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transforms(image)

        percentage = float(item[1])
        patient_id = int(item[2])

        return (image, percentage, patient_id, filename)