import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


DATA_PATH = './Data'
data = []
labels = []
num = 0

for dir in os.listdir(DATA_PATH):
    for img in os.listdir(os.path.join(DATA_PATH, dir)):
        x = image.load_img(os.path.join(DATA_PATH, dir, img), target_size=(250, 250))
        x_array = image.img_to_array(x)
        data.append(x_array)
        labels.append(int(dir))
        

file = open('data.pickle', 'wb')
pickle.dump({'data' : np.asarray(data), 
             'labels' : np.asarray(labels)}, file)
file.close()

