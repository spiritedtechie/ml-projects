import keras
from numpy import ndarray
from sklearn.model_selection import train_test_split

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = (
    keras.datasets.cifar10.load_data()
)

y_labels = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

X_train: ndarray = X_train_raw
X_test: ndarray = X_test_raw
# covert classifications to binary classifications matrix
y_train: ndarray = keras.utils.to_categorical(y_train_raw, 10)
y_test: ndarray = keras.utils.to_categorical(y_test_raw, 10)

# Split the training data into training and validation
split_sets: list[ndarray] = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = split_sets
