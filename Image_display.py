import numpy as np  # forlinear algebra
import matplotlib.pyplot as plt  # for plotting things
import os
from PIL import Image


# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix


Cload = Image.open(r'C:\Users\Administrator\Desktop\Project\datasets\Cancerous\Cancerous20.jpg')
NCload = Image.open(r'C:\Users\Administrator\Desktop\Project\datasets\NonCancerous\Non-Cancerous20.jpg')

# Let's plt these images
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(NCload)
a1.set_title('NonCancerous CT')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(Cload)
a2.set_title('Cancerous CT')
plt.savefig('xx')