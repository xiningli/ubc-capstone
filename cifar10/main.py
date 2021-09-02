import time
# import keras
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential


from tensorflow.keras import datasets, layers, models
from keras.utils import np_utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

# example of loading the cifar10 dataset
from matplotlib import pyplot
from keras.datasets import cifar10
# load dataset
(train_images, train_labels), (test_images, test_labels)= cifar10.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(train_images[i])
# show the figure
pyplot.show()


# one hot encode target values
from keras.utils import to_categorical

# Converting the pixels data to float type
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
 
# Standardizing (255 is the total number of pixels an image can have)
train_images = train_images / 255
test_images = test_images / 255 

# One hot encoding the target class (labels)
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
# Creating a sequential model and adding layers to it
feature_extractor = Sequential()
feature_extractor.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', 
    input_shape=(32,32,3)))
feature_extractor.add(layers.BatchNormalization())

feature_extractor.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
feature_extractor.add(layers.BatchNormalization())
feature_extractor.add(layers.MaxPooling2D(pool_size=(2,2)))
feature_extractor.add(layers.Dropout(0.3))

feature_extractor.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
feature_extractor.add(layers.BatchNormalization())

feature_extractor.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
feature_extractor.add(layers.BatchNormalization())
feature_extractor.add(layers.MaxPooling2D(pool_size=(2,2)))
feature_extractor.add(layers.Dropout(0.5))

feature_extractor.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
feature_extractor.add(layers.BatchNormalization())

feature_extractor.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
feature_extractor.add(layers.BatchNormalization())
feature_extractor.add(layers.MaxPooling2D(pool_size=(2,2)))
feature_extractor.add(layers.Dropout(0.5))

feature_extractor.add(layers.Flatten())
feature_extractor.add(layers.Dense(128, activation='relu'))
feature_extractor.add(layers.BatchNormalization())
feature_extractor.add(layers.Dropout(0.5))


from tensorflow.keras.models import Model
x = feature_extractor.output  
prediction_layer = Dense(num_classes, activation = 'softmax')(x)
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)

from tensorflow.keras.optimizers import Adam, SGD
learning_rate = 0.001
cnn_model.compile(loss="categorical_crossentropy", 
    optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

history = cnn_model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_split=0.1)


pred = cnn_model.predict(test_images)
print(pred)
# Converting the predictions into label index 
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
cnn_f1_score = f1_score(np.argmax(test_labels, axis=1), np.array(pred_classes), average='weighted')
cnn_accuracy = accuracy_score(np.argmax(test_labels, axis=1), np.array(pred_classes))

print("cnn_f1_score: " + str(cnn_f1_score))
print("cnn_accuracy: " + str(cnn_accuracy))


from keras.models import Model
X_for_RF = feature_extractor.predict(train_images) #This is out X input to RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_for_RF, np.argmax(train_labels, axis=1)) #For sklearn no one hot encoding
X_test_feature = feature_extractor.predict(test_images)
prediction_RF = rf.predict(X_test_feature)
from sklearn.metrics import f1_score
cnn_rf_f1=f1_score(np.argmax(test_labels, axis=1), prediction_RF, average='weighted')
cnn_rf_accuracy=accuracy_score(np.argmax(test_labels, axis=1), prediction_RF)

print("cnn_rf_f1_score: " + str(cnn_rf_f1))
print("cnn_rf_accuracy: " + str(cnn_rf_accuracy))

