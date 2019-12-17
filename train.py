import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

'''
Current nesting system:
.
├── training
│   ├── train
│   └── train.csv
└── testing
    ├── test
    └── test.csv

Formatting of training csv file:
filename,label
where filename is the name of the photo and label is an int showing the category

Formatting of testing csv file:
id
where id is the name of them image represented by an integer
'''
# read the train csv file using pandas
train = pd.read_csv('training/train.csv')

# Reading the training images
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(
        'training/train/'+train['filename'][i],
        target_size=(28, 28, 1),
        color_mode="grayscale"
    )
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

# Creating the target variable
y = train['label'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# initialize the model
model = Sequential()

# convolution
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# pooking
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flattening
model.add(Flatten())

# full connection
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
num_classes = 2
model.add(Dense(num_classes, activation='softmax'))

# compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

# fit to photos
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save('best_model.h5py')

# read the test csv file using pandas
test = pd.read_csv('testing/test.csv')

# make sure the test photos are all in jpeg gormat
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(
        'testing/test/'+test['id'][i].astype('str')+'.jpeg',
        target_size=(28, 28, 1),
        color_mode="grayscale"
    )
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

# making predictions
prediction = model.predict_classes(test)

# create a csv file with the predictions for the test files
sample = pd.read_csv('testing/test.csv')
sample['label'] = prediction
sample.to_csv('sample_prediction.csv', header=True, index=False)
