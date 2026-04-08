#!/usr/bin/env python3
"""Module for YOLO v3 object detection using Darknet Keras model."""
@staticmethod
def load_images(folder_path):
    """Load all images from a folder.

    Args:
        folder_path: string representing the path to the folder
            holding all the images to load

    Returns:
        tuple: (images, image_paths)
            images: list of images as numpy.ndarrays
            image_paths: list of paths to the individual images
    """
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ]
    images = [cv2.imread(path) for path in image_paths]
    return images, image_paths
