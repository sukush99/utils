import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Transform: PIL Image -> Tensor
to_tensor = transforms.ToTensor()

def show_images(full_dataset, num_images=10):
    """
    Display a number of images from the dataset.
    
    Args:
        full_dataset (Dataset): PyTorch Dataset or list of (image, label) tuples.
        num_images (int): Number of images to display.
    """
    indices = np.random.choice(len(full_dataset), num_images, replace=False)

    for i in indices:
        img, label = full_dataset[i]
        
        # Ensure the image is a tensor
        if isinstance(img, Image.Image):
            img_tensor = to_tensor(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        print(f"Label: {label}")

        # Convert from tensor to numpy for plotting
        npimg = img_tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        plt.imshow(npimg)
        plt.axis('off')
        plt.show()
