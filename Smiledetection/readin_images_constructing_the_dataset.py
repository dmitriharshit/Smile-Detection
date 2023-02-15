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

pixel = np.array(pixel)

# so we are done with features now we see labels

for name in os.listdir(directory):
    if(name[0:5] == 'happy'):
        labels.append([1, 0])
    if(name[0:3] == 'sad'):
        labels.append([0, 1])

labels = np.array(labels)
# print(labels.shape)

# Applying min-max normalization to features (here just /255)
pixel = pixel/255.0
print(pixel)
