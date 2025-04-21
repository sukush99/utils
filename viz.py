import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Convert PIL Image to Tensor
to_tensor = transforms.ToTensor()

def show_images_grid(dataset, num_images=9, rows=3, cols=3):
    """
    Display a grid of images from a dataset with their class names.

    Args:
        dataset (Dataset): A PyTorch Dataset object like CIFAR-10.
        num_images (int): Total number of images to show.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    assert rows * cols == num_images, "rows * cols must equal num_images"

    indices = np.random.choice(len(dataset), num_images, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for ax, idx in zip(axes.flat, indices):
        img, label = dataset[idx]  

        # Convert image to tensor if it's not already
        if isinstance(img, Image.Image):
            img_tensor = to_tensor(img)
        else:
            img_tensor = img

        # Convert tensor to numpy and rearrange dimensions
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Show image
        ax.imshow(img_np)
        class_name = dataset.classes[label] if hasattr(dataset, 'classes') else str(label)
        ax.set_title(class_name, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()