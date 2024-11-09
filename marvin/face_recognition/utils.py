import config

import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf

def euclidean_distance(vector1, vector2):
    """
    Calculates the euclidean distance between two embedding vectors
    
    Parameters:
    vector1 (vector): The first vector
    vector2 (vector): The second vector
    
    Returns:
    float: The Euclidean distance between vector1 and vector2
    """
    distance = np.sqrt(np.sum((np.array(vector1), np.array(vector2))**2))
    return distance

def create_pairs(positive_dataset_path, negative_dataset_path):
    """
    Creates a dataset for a SNN by generating positive and negative pairs, shuffling them, and
    combining them into a single dataset.

    Parameters:
    positive_dataset_path (str): The filepath to the directory containing positive images
    negative_dataset_path (str): The filepath to the directory containing negative images

    Returns:
    list: A shuffled list of tuples, where each tuple contains two image identifiers and a label
    """

    positive_image_paths = get_image_paths(positive_dataset_path)
    negative_image_paths = get_image_paths(negative_dataset_path)

    positive_pairs = set()
    for img1, img2 in itertools.combinations(positive_image_paths, 2):
        pair = sorted([img1, img2])
        positive_pairs.add((pair[0], pair[1], 1))
        
    negative_pairs = set()
    for img1 in positive_image_paths:
        for img2 in negative_image_paths:
            pair = sorted([img1, img2])
            negative_pairs.add((pair[0], pair[1], 0))
    
    dataset = list(positive_pairs.union(negative_pairs))
    random.shuffle(dataset)
    
    processed_dataset = []
    for image1, image2, label in dataset:
        processed_image1 = preprocess_images(image1, config.IMG_SHAPE)
        processed_image2 = preprocess_images(image2, config.IMG_SHAPE)
        processed_dataset.append((processed_image1, processed_image2, label))
    
    return processed_dataset

def get_image_paths(dataset_path):
    """
    Helper function for create_pairs function, creates list of all images in the directory
    
    Parameters:
    dataset_path (str): The path to the directory with the desired images
    
    Returns:
    list: list of all images inside the directory
    """
    image_paths = []
    files = os.listdir(dataset_path)
    for f in files:
        full_path = os.path.join(dataset_path, f)
        if os.path.isfile(full_path):
            image_paths.append(full_path)
    return image_paths

def preprocess_images(image_path, image_shape):
    """
    Helper function for create_pairs function, resizes image to image size specified in config.py
    
    Parameters:
    image_path (str): Path to the image file.
    img_shape (tuple): Desired image dimensions (height, width, channels)
    
    Returns:
    numpy.ndarray: Preprocessed image array
    """
    dimensions = (image_shape[1], image_shape[0])
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize(dimensions)
    image = np.array(image) / 255.0
    return image

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function.

    Parameters:
    y_true (tensor): True labels (1 for similar, 0 for dissimilar).
    y_pred (tensor): Predicted distances between embeddings.

    Returns:
    tensor: Loss value.
    """
    margin = 1.0
    loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(loss)

def plot_training(training_history, path):
    """
    Creates a visual graph to see how loss function is performing
    
    Parameters:
    training_history ():
    path (str): The directory where we save the plot
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(training_history.history["loss"], label="train_loss")
    plt.plot(training_history.history["val_loss"], label="val_loss")
    plt.plot(training_history.history["accuracy"], label="train_acc")
    plt.plot(training_history.history["val_accuracy"], label="val_acc")
    plt.title("Siamese Network Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(path)