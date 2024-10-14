import config
import numpy as np

def euclidean_distance(vector1, vector2):
    distance = np.sqrt(np.sum((np.array(vector1), np.array(vector2))**2))
    return distance

def create_pairs(dataset, pairs=10):
    data = []