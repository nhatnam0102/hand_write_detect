import numpy as np
import cv2
from tensorflow import keras
import csv
import pandas as pd
from keras import backend as K
from csv import DictWriter

df = pd.read_csv('train.csv')
model = keras.models.load_model("model/mnist.h5", compile=False)

image = cv2.imread(r'C:\Users\0180.UPC\source\repos\DrawText\DrawText\bin\Debug\netcoreapp3.1\uPySock\input\draw.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

image = cv2.GaussianBlur(image, (3, 3), 0)
kernel = np.ones((3, 3), np.uint8)
image = cv2.dilate(image, kernel)
cv2.imshow('img', image)
ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# find contours on image and draw it.
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = [cv2.boundingRect(ctr) for ctr in contours]

bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0] + b[1], reverse=False)
result = ""
if len(bounding_boxes) > 0:
    for idx, rect in enumerate(bounding_boxes):
        # draw rectangle on image using contours
        x, y, w, h = (rect[0], rect[1], rect[2], rect[3])
        roi = image[y:y + h, x:x + w]

        roi_resize = roi.copy()
        if w < 100:
            if h > 100:
                roi_resize = cv2.resize(roi, (0, 0), fx=0.6, fy=0.4)
        elif 100 < w < 200:
            if h < 220:
                roi_resize = cv2.resize(roi, (0, 0), fx=0.4, fy=0.4)
        elif w >= 200:
            roi_resize = cv2.resize(roi, (0, 0), fx=0.3, fy=0.3)
        roi = roi_resize
        roi = np.pad(roi, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
        # if w < h:
        #     roi = cv2.copyMakeBorder(roi, 0, 0, abs((h - w) // 2), abs((h - w) // 2), cv2.BORDER_CONSTANT,
        #                              value=[0, 0, 0])
        # else:
        #     roi = cv2.copyMakeBorder(roi, abs((h - w) // 2), abs((h - w) // 2), 0, 0, cv2.BORDER_CONSTANT,
        #                              value=[0, 0, 0])
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        roi = cv2.dilate(roi, (3, 3))
        roi1 = np.array(roi)
        row1 = roi1.ravel()
        listtt = row1.tolist()
        data_to_append = {}
        print(len(df.columns))
        print(len(listtt))
        data_to_append[df.columns[0]] = '6'
        for i in range(len(listtt)):
            j = i + 1
            data_to_append[df.columns[j]] = listtt[i]
        print(df.columns[0])
        print(data_to_append)
        with open('train.csv','a') as f:
            dic_write = DictWriter(f, fieldnames=list(df.columns))
            dic_write.writerow(data_to_append)
            f.close()
        cv2.imwrite('roi.png', roi)
        # reshape your image according to your model
        roi = roi.reshape(-1, 28, 28, 1)

        roi = np.array(roi, dtype='float32')
        roi /= 255
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # to perform prediction on your image
        pred_array = model.predict(roi)
        pred_array = np.argmax(pred_array)
        str_pred = str(pred_array)
        result = result + str_pred
        # print(result)
