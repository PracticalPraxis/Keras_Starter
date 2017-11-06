#This model is trained on a dataset of information about housing in 1970's Boston
#The model is built around the core present in other models in the Keras_Starter folder, with the addition of K-Fold Validation 
#This use of K-Fold Validation makes it somewhat reliable (ultimate predictions are off by about $2,500), but computationally expensieve
#This version of the model also runs for 500 epochs but starts overfitting around the 100th. All in all, an educational experiment

#There are 13 features for each data point, including things like the per capita crime rate for the area, the surrounding pupil-teacher ratio 
#and the average number of rooms per house
#Our model will learn the prices associated with various houses that all have values for each of these features, and then try 
#to predict the median price of a seperate set of homes
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#Arranges the housing data into distinct training and test partitions with analgous divisions
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#Ensures our training and test data is easier to iterate over by making the nominal range of the data much smaller using using feature-wise normalization
#This is important as there are some features in our dataset that use a very wide scale and some that use a very slim scale
#This normalization will ensure they all have similar values
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

#Defines the model we'll be using
def build_model():
	#Declares the overall structure of the model, in this case the Sequential class. The model itself is made out of 3 Dense layers
	model = models.Sequential()
	#In the first layer the model outputs a vector with 64 dimensions, while the relu activation meeans the layer will outputthe maximum value of layer's dot product computation
	model.add(layers.Dense(64, activation='relu',
						   input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	
	#The model is complied with the optimizer rmsprop which which keeps a moving average over the root mean squared gradients (rms) which the gradient at use gets divided by
	#This value is used to reduce our loss value
	#Loss is calculated using Mean Squared Error (MSE), the square of the difference between the model's output and its targets
	#Finaly, we declare a single metric to print, the value of the Mean Absolute Error at each iteration
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

#To improve the accuracy of the model we'll use K-fold validation which splits the data into K equal parts and then trains a model on the reamining K - 1 partitions
#The ultimate value we get is the mean of all the resulsts of these K models
k = 4
#Having declared the relevant variable for K-Fold Validation, we now know our model will run on the below code 4times, for 500 epochs at a time, before finishing
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
#Starting the K-fold process
for i in range(k):
	print('processing fold #', i)
	#Prepares the validation data from pariton k
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	
	#Prepares the training data for all other partitions
	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples],
		 train_data[(i + 1) * num_val_samples:]],
		axis=0)
	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples],
		 train_targets[(i + 1) * num_val_samples:]],
		axis=0)
	
	#Calls the pre-compiled Keras model
	model = build_model()
	#Trains the model on our the data we prepared above
	history = model.fit(partial_train_data, partial_train_targets,
						validation_data=(val_data, val_targets),
						epochs=num_epochs, batch_size=1, verbose=0)
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)
	
average_mae_history = [
	np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#Plots how far our models predictions were in a nice clean graph
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#The data is a bit noisy so let's smooth things out slightly 
def smooth_curve(points, factor=0.9):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()