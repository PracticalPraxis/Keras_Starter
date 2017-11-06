#This model uses the reuters dataset from Keras which is a collection of ~10000 articles covering 
#46 different topics from the Reuters News Agency
#Our model learns by scanning the articles and their associqated topic label
#The model is then shown the validation set of articles and must predict the labels of that dataset
#Overall, this model achieves roughly 77% accuracy in the prediction task, before overfitting past the 9th epoch

#Most eveything used in this program has been repurposed from Keras_IMDB.py, if there is a part of the program 
#you'd like to have explained please refer to that file

from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

#As with Keras_IMDB.py, we seperate the data and restrict our samples to contain only the 
#10000 most frequently used words in order to make training the model less intensive
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#We use one-hot encoding on our labels, with 46 dimensions as there are 46 different topic classifications
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
			  #We use categorical_crossentropy as our loss value as binary_crossentropoy is unsuitable due to 
			  #the n > 2 dimension size of our labels
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
					partial_y_train,
					epochs=9,
					batch_size=512,
					validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()