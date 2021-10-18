import numpy as np
import tensorflow, os, json, pickle, yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from dvclive.keras import DvcLiveCallback
OUTPUT_DIR = "output"
fpath = os.path.join(OUTPUT_DIR, "data.pkl")
with open(fpath, "rb") as fd:
    data = pickle.load(fd)
(x_train, y_train),(x_test, y_test) = data
    
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)

image_size = x_train.shape[1]
input_size = image_size * image_size

params = yaml.safe_load(open("params.yaml"))["train"]
batch_size = params["batch_size"]
hidden_units = params["hidden_units"]
dropout = params["dropout"]
num_epochs = params["num_epochs"]
lr = params["lr"]
conv_activation = params["conv_activation"]

# Model specific code
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255
model = Sequential()
model.add(Conv2D(filters=28, kernel_size=(3,3), activation=conv_activation))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels, activation='softmax'))
# End of Model specific code

opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])

model.fit(
    x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1,
    callbacks=[DvcLiveCallback(os.path.join(OUTPUT_DIR, "model.h5"))])
