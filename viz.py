import numpy as np
import matplotlib.pyplot as plt

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
        print(f"Label: {label}")
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
   
