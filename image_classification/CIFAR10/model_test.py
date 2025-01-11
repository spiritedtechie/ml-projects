import numpy as np
from numpy import ndarray
import keras
from data import X_test, y_test, y_labels
import matplotlib.pyplot as plt
import random
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path


### Load the saved model (run model_train first)
p = Path(__file__).parent.joinpath("models").joinpath("final_model.keras")
model: keras.Model = keras.models.load_model(p.absolute())


### Check model accuracy on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Trained model - test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}")


### Pick a random test image and check prediction
idx = random.randint(0, len(X_test))
image = X_test[idx]
y_pred = y_labels[np.argmax(model.predict(image.reshape(1, 32, 32, 3)))]
print(f"Model predicts that image is {y_pred}")
plt.imshow(image)
plt.show()


### Accuracy confusion matrix on the test data comparing predicted versus actual
print("Predicting test data set")
y_pred: ndarray = model.predict(X_test)
y_pred_unencoded = np.argmax(y_pred, axis=1)
y_test_unencoded = np.argmax(y_test, axis=1)

print("Confusion matrix of predicted versus actual for test data set")
cm = confusion_matrix(y_test_unencoded, y_pred_unencoded)
con = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)

fig, ax = plt.subplots(figsize=(10, 10))
con = con.plot(xticks_rotation="vertical", ax=ax, cmap="cool")
plt.show()
