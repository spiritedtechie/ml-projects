import matplotlib.pyplot as plt
from data import X_train, y_train, X_val, y_val, X_test, y_test, y_labels
import numpy as np


# Understand the shape of the data
print(f"X_train: {X_train.shape}")  # 40000, 32, 32, 3
print(f"y_train: {y_train.shape}")  # 40000, 10
print(f"X_val: {X_val.shape}")  # 10000, 32, 32, 3
print(f"y_val: {y_val.shape}")  # 10000, 10
print(f"X_test: {X_test.shape}")  # 10000, 32, 32, 3
print(f"y_test: {y_test.shape}")  # 10000, 10


# Sample and plot first 25 image in training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    x = X_train[i]
    y = y_labels[np.argmax(y_train[i])]
    plt.imshow(x, cmap=plt.cm.binary)
    plt.xlabel(y)

plt.show()
