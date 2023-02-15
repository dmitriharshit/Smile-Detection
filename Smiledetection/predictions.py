from PIL import Image
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

directory = r"D:\Desktop\ML\Smiledetection\smilesdataset\trainingset"

# these are lists. We will have to convert them to arrays later
pixel = []
# using one_hot_encoding we use happy(1,0) and sad(0,1)
labels = []

# we iterate through the folder 'training dataset' using os.listdir function:
for filename in os.listdir(directory):
    # print(filename) #to see that filename actually ietratess through all the files in the folder
    image = Image.open(directory+'\\'+filename).convert('1')
    pixel.append(list(image.getdata()))
    if(filename[0:5] == 'happy'):
        labels.append([1, 0])
    if(filename[0:3] == 'sad'):
        labels.append([0, 1])

pixel = np.array(pixel)

labels = np.array(labels)

# Applying min-max normalization to features (here just /255)
pixel = pixel/255.0

# model:
model = Sequential()
model.add(Dense(1024, input_dim=1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
# output layer:
model.add(Dense(2, input_dim=128, activation='softmax'))

optimize = Adam(lr=0.005)

model.compile(loss='categorical_crossentropy', optimizer=optimize,
              metrics=['accuracy'])

model.fit(pixel, labels, epochs=1000, batch_size=20, verbose=2)

pixel_test = []
test_image = Image.open(
    r"D:\Desktop\ML\Smiledetection\smilesdataset\testset\happy_test.png").convert('1')
pixel_test.append(list(test_image.getdata()))
pixel_test = np.array(pixel_test)
print(model.predict(pixel_test))
