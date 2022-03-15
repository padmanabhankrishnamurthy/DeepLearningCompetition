import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def norm_tensor_to_img(image, label=None):
    inv_normalize = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    image = inv_normalize(image)
    image = torch.permute(image, [1, 2, 0])
    image = image.numpy() * 255
    image = image.astype(np.uint8)

    plt.imshow(image)
    if label:
        plt.title(label)

    plt.show()