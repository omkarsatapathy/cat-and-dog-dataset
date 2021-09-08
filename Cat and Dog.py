import numpy as np 
import matplotlib.pyplot as plt
import os 
import cv2
from tensorflow.python.keras.layers.core import Dropout, Flatten

datadir = "D:\cat_dog"
catagories = ["dog", "cat"]

for catagory in catagories:
    path = os.path.join(datadir,catagory)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img_array, cmap='gray')
        #plt.show()
        break
    break


new_array = cv2.resize(img_array, (100,100))
plt.imshow(new_array, cmap='gray')
#plt.show()
training_data = []
def create_training_data():
    for catagory in catagories:
        path = os.path.join(datadir,catagory)
        class_num = catagories.index(catagory)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (100, 100))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass    
create_training_data()


import random 
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X=[]
y=[]
for features, label in training_data:
    X.append(features)
    y.append(label)
X =np.array(X).reshape(-1, 100,100, 1)
y = np.array(y)

import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in = open("X.pickle", "rb")
X=pickle.load(pickle_in)


import tensorflow as tf
from tensorflow import keras

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = X/255.0

model = keras.Sequential([
    keras.layers.Conv2D(128, (3,3),activation= 'relu',  input_shape = X.shape[1:]),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3),activation= 'relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=7, verbose=1)



model_json_catsanddogs = model.to_json()
with open ('model.json', 'w') as json_file:
    json_file.write(model_json_catsanddogs)
    
model.save_weights('model_catsanddogs.h5')
