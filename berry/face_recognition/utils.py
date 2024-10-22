import config
import itertools
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(vector1, vector2):
    distance = np.sqrt(np.sum((np.array(vector1), np.array(vector2))**2))
    return distance

def create_pairs(dataset, pairs=10):
    positive_pairs = []
    for img1, img2 in itertools.combinations(config.POSITIVE_DATASET_PATH, 2):
        pair = (img1, img2, 1)
        positive_pairs.append(pair)
    seen_pairs = {}
    unique_positive_pairs = []
    for img1, img2, label in positive_pairs:
        pair_id = tuple(sorted([id(img1), id(img2)]))
        if pair_id not in seen_pairs:
            seen_pairs.add(pair_id)
            unique_positive_pairs.append((img1, img2, label))
    positive_pairs = unique_positive_pairs
        
    negative_pairs = []
    for img1 in config.POSITIVE_DATASET_PATH:
        for img2 in config.NEGATIVE_DATASET_PATH:
            pair = (img1, img2, 0)
            negative_pairs.append(pair)
    seen_pairs = {}
    unique_negative_pairs = []
    for img1, img2, label in negative_pairs:
        pair_id = tuple(sorted([id(img1), id(img2)]))
        if pair_id not in seen_pairs:
            seen_pairs.add(pair_id)
            unique_negative_pairs.append((img1, img2, label))
    negative_pairs = unique_negative_pairs
            
    return positive_pairs, negative_pairs

def plot_training(training_history, path):
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