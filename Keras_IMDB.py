#This is the first fully operational program I wrote using Keras, so there is heavy commenting throughout. Hope it's helpful!
#Model achieves a 65% - 70% validation accuracy, starts overfitting around the 4/20 epoch

#The data itself: 50,000 highly polarizing movie reviews from IMDB - 25,000 positive reviews and 25,000 negative reviews
#These reviews themselves have been reduced to collections of integers, where each word of the review is represented by a number
#e.g. 'great' as 147 and 'fly' as 861
#So a review stating "This movie was so great it made me fly" would look like this in our dataset: 
#[4, 1, 67, 998, 147, 76, 98, 32, 861] ([] used for neatness and not necessarily to present an array)
#The label is just a sentiment score of 1 or a 0, marking a given review as either positive or negative
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

#Splitting the loaded data into separate categories for training and testing (we won't be using the test data, but it still needs 
#to be loaded separately from the training data)
#Limiting this program to the 10000 most frequent words in the dataset for efficiency's sake
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#Defines the method by we'll use to transform the imdb integer lists into vectors. In practice this looks like the following:
#The review "This movie sucked" is represented by our dataset as [4, 1, 99]
#we translate the [4, 1, 99] list into a format that is intelligible to our model by creating a vector of N dimensions 
#(where N is the highest number presented in our dataset)
#We then assign the 17th, 4th and 99th entry in the vector a value of 1 and all spaces with 0
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

#Our training data and labels gets vectorised
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')

#Declares a Sequential model with 3 Dense layers
#Sequential is the most basic kind of overall structure for ML models, where each layer does its job fully, before 
#passing the complete output to the next layer
#This is not the case for the more advanced models out there, which incorporate complex structures with non-sequential 
#communication between layers
model = models.Sequential()
#The Dense layer is a very basic neural network layer that lacks the bells or whistles of more complex layers
#It essentially computer the dot product of the input and the given weights before performing an activation on this dot product
#The activation here is 'relu' which just takes the max value of the layer's computation and outputs it
#Unintuitively, the output shape is given first (16) and the input shape is given last
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
#Finally we end with an output of 1 and the sigmoid activation to generate a probability in the [0, 1] interval
model.add(layers.Dense(1, activation='sigmoid'))

#We compile the model, priming it to be used on our dataset
#First we call the optimizer, which will alter our model's operation based on the loss value it receives
#In this case we're using the rmsprop optimizer which keeps a moving average over the root mean squared gradients (rms) 
#which the gradient at use gets divided by
#Secondly, we're using the binary variant of crossentropy loss calculation to measure our loss due to the nature of our data 
#i.e. we're sorting the review into positive or negative, so we need a loss measurement sensitive to binary classification 
#and probability distribution (hence, crossentropy)
#Finally, we're measuring the overall accuracy of our model by calling the accuracy metric
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Seperating our validation data from our training data, before and after the 10000th sample respectivley
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#We call the model on our data using model.fit
#First we feed it our actual data, the words of the review (partial_x_train) and the 
#sentiment score for the review (partial_y_train)
#The model will run through all the words and labels it gets, and learn the score associated with certain reviews
#It will then be given reviews it does not know the sentiment score of (our validation set) and try to guess 
#their sentiment score
#We tell the model to iterate over this process 20 times (epochs) in batches of 512 reviews
#Finally, we specify our validation to be the 10000 samples we've previously set aside
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#Assiging the model's various outputs to discrete variables so they can be plotted
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#Plotting the training and validation loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Plotting the training and validation accuracy
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
