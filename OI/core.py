
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import seaborn as sns
import time
import squeezenet


def data_preprocessing(model_type):
	# load the MNIST dataset
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# flatten 28*28 images to a 784 vector for each image
	num_pixels = X_train.shape[1] * X_train.shape[2]
	print("Num pixels: ", num_pixels)
	if model_type == "base":
		X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
		X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
	if model_type == "lstm":
		X_train = np.reshape(X_train, (X_train.shape[0], 1, num_pixels))
		X_test = np.reshape(X_test, (X_test.shape[0], 1, num_pixels))
	if model_type != "base" and model_type != "lstm":
		print ("Wrong model type... exiting.")
		exit()
	# normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255
	# one hot encode outputs
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]
	return num_classes, num_pixels, X_test, X_train, y_test, y_train

batch_size = 200

# define sequential dense model
def base_model():
	#pre-process input data
	num_classes, num_pixels, X_test, X_train, y_test, y_train = data_preprocessing("base")
	# create model
	start_time = time.time()
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model, start_time, X_test, X_train, y_test, y_train

# define LSTM model
def lstm_model():
	#pre-process input data
	num_classes, num_pixels, X_test, X_train, y_test, y_train = data_preprocessing('lstm')
	# create model
	start_time = time.time()
	print(start_time)
	model = Sequential()
	model.add(LSTM(num_pixels, input_shape=(1, num_pixels), activation='relu', return_sequences=True))
	model.add(LSTM(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model, start_time, X_test, X_train, y_test, y_train

# build the model
model, start_time, X_test, X_train, y_test, y_train = lstm_model()

model = SqueezeNet(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
print("time: ",time.time())
print("Learning took ", time.time() - start_time, " seconds")

# Final evaluation of the model
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
score, acc = model.evaluate(X_test, y_test, batch_size=200)
print("Train set accuracy: ", history.history['accuracy'][-1])
print("Test set  accuracy: ", acc)

# Plot loss function over training 
fig, ax = plt.subplots()
ax.plot(range(1,11), history.history['loss'], label='Train Loss')
ax.plot(range(1,11), history.history['val_loss'], label='Validation Loss')
ax.legend(loc='best')
ax.set(xlabel='epochs', ylabel='accuracy')

# Plot Confusion matrix for train set
fig = plt.figure(figsize=(10, 10)) # Set Figure

y_pred_train = model.predict(X_train) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Y_pred_train = np.argmax(y_pred_train, 1) # Decode Predicted labels
Y_train = np.argmax(y_train, 1) # Decode labels

mat_train = metrics.confusion_matrix(Y_train, Y_pred_train) # Confusion matrix
sns.heatmap(mat_train.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.title("Confusion Matrix Train set")
plt.xlabel('Predicted Values')
plt.ylabel('True Values for train dataset')

# Plot Confusion matrix
fig = plt.figure(figsize=(10, 10)) # Set Figure

y_pred = model.predict(X_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels
Y_test = np.argmax(y_test, 1) # Decode labels

mat = metrics.confusion_matrix(Y_test, Y_pred) # Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.title("Confusion Matrix Test set")
plt.xlabel('Predicted Values')
plt.ylabel('True Values for test dataset')

wrong_predictions = []
wrong_img = []
ground_truth = []
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)
print(X_test.shape)
for i in range(X_test.shape[0]):
	if Y_pred[i] != Y_test[i]:
		wrong_predictions.append(Y_pred[i])
		wrong_img.append(X_test__[i])
		ground_truth.append(Y_test[i])

fig, axis = plt.subplots(3, 3, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for k, ax in enumerate(axis.flat):
	ax.imshow(wrong_img[k], cmap='binary')
	ax.set(title = f"Real Number is {ground_truth[k]}\nPredict Number is {wrong_predictions[k]}")
plt.show()

#https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
#https://github.com/techedlaksh/MNIST-keras/blob/master/keras.MNIST.ipynb