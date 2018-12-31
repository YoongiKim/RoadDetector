import input_pipeline
import MyKeras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import cv2
import way
import random

train_gen, valid_gen = input_pipeline.get_generators(batch_size=1)

model, epoch = MyKeras.load_latest_model('main')

for x, y in valid_gen:
    y_hat = model.predict(x)
    y_hat = np.argmax(y_hat, axis=-1)
    y_hat = np.array(to_categorical(y_hat, 3)[0] * 255.0).astype(np.uint8)
    y_hat[:,:,0] = 0
    map = y_hat[:,:,1]

    plt.imshow(np.array(x[0]*255.0 + y_hat/2.0).astype(np.uint8))
    plt.show()

    # best_a = way.find_way(map)
    # print(best_a)
    input()
