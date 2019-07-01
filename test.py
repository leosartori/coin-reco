from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

#from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import preprocess_input

import os
import csv

INPUT_SIZE = 224
TRAIN_PATH = 'images/coins-dataset/classified/train'

MODEL_FILENAME = 'inc_model2019-07-01T18:45:56.878010'
PREDICT_PATH = "images/detect"

# getting list of labels
# todo: remove global variable?
LABELS = [i for i in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH,i))]
LABELS.sort()
print(LABELS)

# load array of images to predict
#x_predict = []
#files = glob.glob(PREDICT_PATH + '/*')
#for myFile in files:
#    print('Image: ' + str(myFile))

#    image = cv2.imread(myFile)
    # todo : remove global variable?
#    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
#    image = np.moveaxis(image, -1, 0)
#    x_predict.append(image)

# check for array shape
#print('X_data shape:', np.array(x_predict).shape)

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

#predictions = model.predict(np.array(x_predict), batch_size=None, verbose=1)

# todo: learn about keras.utils.Sequence
filenames = predict_generator.filenames
nb_samples = len(filenames)
predictions = model.predict_generator(predict_generator, steps=nb_samples, verbose=1)

print('Predictions: ' + str(predictions))

# print predicted label

pred_index = predictions.argmax(axis=-1)
pred_label = [LABELS[index] for index in pred_index]
print('Predicted labels: ' + str(pred_label))

with open('images/detect/pred.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(pred_label)
csvFile.close()