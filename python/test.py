from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import model_from_json

import os
import cv2
import glob
import numpy as np
import train

MODEL_FILENAME = 'model2019-06-29T22:53:13.260338'
PREDICT_PATH = "../images/predict/*"

# getting list of labels
# todo: remove global variable?

LABELS = ['1e', '2e', '20c', '50c', 'unknown']
#LABELS = [i for i in os.listdir(train.TRAIN_PATH) if os.path.isdir(i)]
#LABELS.sort()
#print(LABELS)

# load array of images to predict
x_predict = []
files = glob.glob(PREDICT_PATH)
for myFile in files:
    # print('Image: ' + str(myFile))

    image = cv2.imread(myFile)
    # todo : remove global variable?
    image = cv2.resize(image, (train.INPUT_SHAPE, train.INPUT_SHAPE))
    image = np.moveaxis(image, -1, 0)
    x_predict.append(image)

# check for array shape
print('X_data shape:', np.array(x_predict).shape)

# load pretrained model
json_file = open(MODEL_FILENAME + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(MODEL_FILENAME + ".h5")
print("Loaded model from disk")

# make predictions
predictions = model.predict(np.array(x_predict), batch_size=None, verbose=1)
# TODO: try model.predict_classes

print('Predictions: ' + str(predictions))

# print predicted label
pred_index = predictions.argmax(axis=-1)
pred_label = [LABELS[index] for index in pred_index]
print('Labels: ' + str(pred_label))