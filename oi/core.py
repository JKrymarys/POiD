import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



#https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

image_size=(200, 200)
# Create a datasets and scale them to image_size
train_ds = keras.preprocessing.image_dataset_from_directory(
  'chest_xray/train', batch_size=64, image_size=image_size)

val_ds = keras.preprocessing.image_dataset_from_directory(
  'chest_xray/val', batch_size=64, image_size=image_size)

test_ds = keras.preprocessing.image_dataset_from_directory(
  'chest_xray/test', batch_size=64, image_size=image_size)



# Build a  model
inputs = keras.Input(shape=(200, 200,3))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()


model.compile(optimizer='adam',
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],)



epochs = 5
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

fig, ax = plt.subplots()
ax.plot(range(1,6), history.history['loss'], label='Train Loss')
ax.plot(range(1,6), history.history['val_loss'], label='Validation Loss')
ax.legend(loc='best')
ax.set(xlabel='epochs', ylabel='loss')


print("Test data")
loss, acc = model.evaluate(test_ds)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

plt.show()



# ---------------------- CONFUSION MATRIX --------------------

# Y_train_prediction = model.predict(x_train)
# y_train_labels = np.argmax(Y_train_prediction, axis=1)
# Y_train = np.argmax(y_train, 1)
# print('Confusion Matrix')
# print(confusion_matrix(Y_train, y_train_labels))


# ---------------------- WRONG PREDICTIONS --------------------

# wrong_predictions = []
# wrong_img = []
# ground_truth = []
# X_test__ = x_test.reshape(x_test.shape[0], 28, 28)
# y_test_prediction = model.predict(x_test)
# y_test_labels = np.argmax(y_test_prediction, axis=1)
# Y_test = np.argmax(y_test, 1)
# print(x_test.shape)
# for i in range(x_test.shape[0]):
#     if y_test_labels[i] != Y_test[i]:
#       wrong_predictions.append(y_test_labels[i])
#       wrong_img.append(X_test__[i])
#       ground_truth.append(Y_test[i])

# fig, axis = plt.subplots(3, 3, figsize=(10, 10))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# for k, ax in enumerate(axis.flat):
#  	ax.imshow(wrong_img[k], cmap='binary')
#  	ax.set(title = f"Real number: {ground_truth[k]}\nPredicted number: {wrong_predictions[k]}")
# plt.show()
