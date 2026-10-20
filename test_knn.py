from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

# load dataset
iris = load_iris()
X = iris.data   # features (sepal/petal lengths and widths)
y = iris.target # labels (species: 0, 1, or 2)

print("iris data:  ", X.shape, X.dtype)
print("iris labels:", y.shape, y.dtype)