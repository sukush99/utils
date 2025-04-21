import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_images(full_dataset, num_images):
    """
    Display a grid of images from the dataset.
    
    Args:
        full_dataset (list): List of images to display.
        num_images (int, optional): Number of images to display. Defaults to 10.
    """


    indices = np.random.choice(len(full_dataset), num_images, replace=False)
    for i in indices:
        img, label = full_dataset[i]
        img_tensor = to_tensor(img)
        print(f"Label: {label}")
        npimg = img_tensor.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
   
