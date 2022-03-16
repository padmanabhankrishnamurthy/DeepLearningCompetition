import sys
sys.path.insert(0, '..')

import torch 
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from data_utils.misc import norm_tensor_to_img
from data_utils.SingleImageDataset import SingleImageDataset
from models.SingleImageModel import SingleImageModel

def train(model, train_loader, optimizer, criterion, device, epochs):
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader)

        for batch in pbar:
            # every item in batch: (image, percentage, patient_id, filename)
            optimizer.zero_grad()

            images = batch[0]
            percentages = batch[1]

            predictions = model(images.to(device))
            batch_loss = criterion(percentages.to(device), predictions)
            epoch_loss+=batch_loss

            batch_loss.backward()
            optimizer.step()
            pbar.set_description('Epoch: {} Loss: {}'.format(epoch + 1, batch_loss))

        print(f"Avg Loss: {epoch_loss/len(train_loader)}\n")

if __name__ == '__main__':
    train_file = '../data/Train.csv'
    image_dir = '../data/Train'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SingleImageModel(classifier=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    batch_size = 4
    epochs = 10

    dataset = SingleImageDataset(train_file=train_file, image_dir=image_dir, onehot=True)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs
    )


