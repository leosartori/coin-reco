#######################################################################

# Coin Detector and Recognizer - Exam of Computer Vision 2018/2019 UNIPD

# @author Leonardo Sartori (leonardo.sartori.1@studenti.unipp.it)
# @version 1.0

#######################################################################

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

import os
import csv

INPUT_SIZE = 224

MODEL_FILENAME = 'model2019-07-02T23:21:44.710661'
PREDICT_PATH = "images/detect"

# getting list of labels
LABELS = ['10c', '1c', '1e', '20c', '2c', '2e', '50c', '5c', 'unk']
LABELS.sort()
print(LABELS)

# generator to preprocess and feed images as set in the network
predict_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

predict_generator = predict_datagen.flow_from_directory(
    PREDICT_PATH,  # this is the target directory
    target_size=(INPUT_SIZE, INPUT_SIZE),  # all images will be resized to 150x150
    batch_size=1,
    class_mode='categorical'
)

# load pretrained model
json_file = open(MODEL_FILENAME + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(MODEL_FILENAME + ".h5")
print("Loaded model from disk")

# make predictions
filenames = predict_generator.filenames
nb_samples = len(filenames)
predictions = model.predict_generator(predict_generator, steps=nb_samples, verbose=1)

# print percentages
print('Predictions: ' + str(predictions))

# print predicted label
pred_index = predictions.argmax(axis=-1)
pred_label = [LABELS[index] for index in pred_index]
print('Predicted labels: ' + str(pred_label))

# save results as CSV, to be open in main.cpp
with open('images/detect/pred.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(pred_label)
csvFile.close()
