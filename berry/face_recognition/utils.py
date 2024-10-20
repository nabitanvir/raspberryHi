import config
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(vector1, vector2):
    distance = np.sqrt(np.sum((np.array(vector1), np.array(vector2))**2))
    return distance

def create_pairs(dataset, pairs=10):
    data = []

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