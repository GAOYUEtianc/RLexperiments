
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims

from keras.models import Model
from keras.models import load_model

from keras.optimizers import Adam

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# Import data
(X_train, y_train), (X_test, y_test) =  mnist.load_data()
# Data Preparation
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Cut the image in X_test vertically in half 
def CutHalf(x):
    height,width = x.shape
    return [x[: , :width//2] , x[:, width//2:] ]
def PasteTogether(x_1, x_2):
    height, width = x_1.shape
    x = np.zeros((height, width*2))
    x[:,width//2] = x_1
    x[:, width//2:] = x_2
    return x
# Sort X_train according to labels
index_label = np.argsort(y_train)
X_train_sorted = X_train[index_label]
y_train_sorted = y_train[index_label]

X_train_half = np.zeros((2*len(X_train_sorted),28,14))
for i in range(0,2*len(X_train),2):
    X_train_half[i], X_train_half[i+1] = CutHalf(X_train_sorted[i//2])

X_train_later = []
# Generate fake images by pasting unmatched figures
for i in range(10):
    # Inverse left and right for images of label i
    for j in range(0, 1200, 2):
        X_train_later.append(np.hstack((X_train_half[12000*i+j+1],X_train_half[12000*i+j])))
    # collapse s.t. left is of label i, right is of another label
    exclude_i = np.delete(np.arange(10), i)
    for label in exclude_i:
        for j in range(0,1200, 2):
            X_train_later.append(np.hstack((X_train_half[12000*i+j],X_train_half[12000*label+j+1])))
X_train_later  = np.array(X_train_later)

# Cut the test set into half pieces, and shuffle them
X_test_half = np.zeros((2*len(X_test),28,14))
for i in range(0,2*len(X_test),2):
    X_test_half[i], X_test_half[i+1] = CutHalf(X_test[i//2])
# Shuffle the half pieces
np.random.shuffle(X_test_half)

# Combine the generated fake images with the original training dataset
X_train_double = np.concatenate((X_train, X_train_later), axis=0)
X_train_double = expand_dims(X_train_double, axis=-1)
print(X_train_double.shape)

X_train = expand_dims(X_train, axis = -1)
#X_test = expand_dims(X_test, axis = -1)
# The first half dataset is true and labeled 1 , the later half (fake data) is labeled 0
y_train_binary = np.zeros(2*len(X_train))
for i in range(60000):
    y_train_binary[i] = 1

# Train the model and save
# The discriminator model takes input an image and judge whether its a true image or not
kernel_size = (3,3)
strides_size = (2,2)
input = Input(shape = (28,28,1))
# First Layer
Encoder = Conv2D(128, kernel_size, strides_size, padding = 'same')(input)
Encoder = LeakyReLU(alpha=0.2)(Encoder)
# Second Layer
Encoder = Conv2D(128, kernel_size, strides_size, padding = 'same')(Encoder)
Encoder = LeakyReLU(alpha=0.2)(Encoder)
# Third Layer
Encoder = Conv2D(128, kernel_size, strides_size, padding = 'same')(Encoder)
Encoder = LeakyReLU(alpha=0.2)(Encoder)
# Feature maps 
Encoder = Flatten()(Encoder)
# Dropout features
Encoder = Dropout(0.4)(Encoder)
# Output layer is of dim 1 
outputLayer = Dense(1, activation='sigmoid')(Encoder)
# The discriminator model : input (28X28X1), output layer is dim 1
discriminator = Model(input, outputLayer)
discriminator.compile(loss='binary_crossentropy', optimizer = Adam(learning_rate = 0.001, beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam"))

# Train for 10 epochs
discriminator.fit(X_train_double, y_train_binary, epochs=10, batch_size=1000, verbose=2, validation_split=0.1)
path = "/Users/GYOK1678/Desktop/HalfImages/discriminator_model"
discriminator.save(path)

