import numpy as np
import cv2
from tensorflow import keras
from keras import backend as K


class Model:

    @staticmethod
    def load_model():
        K.clear_session()
        model = keras.models.load_model("model/csv_model.h5", compile=False)
        return model

    @staticmethod
    def recognize(path, model):
        # load your image to recognize image
        image = cv2.imread(path)
        image1 = cv2.imread(path)

        # perform some basic operation to smooth image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find threshold image
        ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        image = cv2.GaussianBlur(image, (3, 3), 0)
        kernel = np.ones((3, 3), np.uint8)

        image = cv2.dilate(image, kernel)
        ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find contours on image and draw it.
        contours, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(ctr) for ctr in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0] + b[1], reverse=False)
        result = ""
        for idx, rect in enumerate(bounding_boxes):
            # draw rectangle on image using contours
            # # resize image
            cv2.rectangle(image1, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
            roi = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            roi = np.pad(roi, (20, 20), 'constant', constant_values=(0, 0))

            roi = cv2.copyMakeBorder(roi, 0, 0, abs((rect[3] - rect[2]) // 2), abs((rect[3] - rect[2]) // 2),
                                     cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

            # reshape your image according to your model
            roi = roi.reshape(-1, 28, 28, 1)
            roi = np.array(roi, dtype='float32')
            roi /= 255

            # to perform prediction on your image
            pre_array = model.predict(roi)
            pre_array = np.argmax(pre_array)
            str_pre = str(pre_array)
            result = result + str_pre

            # print result
            print('Result {0}: {1}'.format(idx, pre_array))

            # print text on your image
            cv2.putText(image1, str(pre_array), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
            # show your image

        # print(result)
        # f = open("output/result.txt", "w")
        # f.write(result)
        cv2.imwrite('output/res_img.png', image1)
        return result
