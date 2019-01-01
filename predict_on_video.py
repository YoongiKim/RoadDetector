import input_pipeline
import MyKeras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import cv2
import way
import random
from tqdm import tqdm

model, epoch = MyKeras.load_latest_model('models/main')

vidcap = cv2.VideoCapture('video/1.mp4')
total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video/out.mp4', fourcc, 60, (1280, 720))

success, image = vidcap.read()
for i in tqdm(range(0, total)):
    if success:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 288))
        img = np.array([img]) / 255.0

        y_hat = model.predict(img)
        y_hat = np.argmax(y_hat, axis=-1)
        y_hat = np.array(to_categorical(y_hat, 3)[0] * 255.0).astype(np.uint8)
        y_hat[:, :, 0] = 0

        add = np.array(img[0]*255.0 + y_hat/2.0).astype(np.uint8)
        add = cv2.cvtColor(add, cv2.COLOR_RGB2BGR)

        out.write(add)  # save video frame
        cv2.imshow('result', add)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            exit(1)

        success, image = vidcap.read()

