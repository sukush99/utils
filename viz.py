import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the transform once
to_tensor = transforms.ToTensor()

def show_images(full_dataset, num_images=10):
    """
    Display a grid of images from the dataset.
    
    Args:
        full_dataset (Dataset): PyTorch Dataset object.
        num_images (int): Number of images to display.
    """

    indices = np.random.choice(len(full_dataset), num_images, replace=False)
    for i in indices:
        img, label = full_dataset[i]
        
        # Convert to tensor if not already
        if isinstance(img, Image.Image):
            img_tensor = to_tensor(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        print(f"Label: {label}")
        npimg = img_tensor.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
