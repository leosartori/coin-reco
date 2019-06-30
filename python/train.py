import datetime
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD

from keras import backend as K
# TODO: investigate ordering in dimensions (to work now it set as Theano, channel first: (3,150,150))
K.set_image_dim_ordering('th')

INPUT_SHAPE = 150
TRAIN_PATH = '/home/leonardo/Documents/computer_vision/project/images/total_denis/'

BATCH_SIZE = 4

# todo: remove this
labels_num = 5

def create_model(l_num):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, INPUT_SHAPE, INPUT_SHAPE)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(l_num))
    model.add(Activation('softmax'))

    return model

def create_model_keras(l_num):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, INPUT_SHAPE, INPUT_SHAPE)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(l_num, activation='softmax'))
    return model

if __name__ == '__main__':

    # TODO: fix this
    #labels_num = sum(os.path.isdir(i) for i in os.listdir(TRAIN_PATH))
    print ("labels_num: " + str(labels_num))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = create_model_keras(labels_num)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # todo: reactivate this
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    # test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of path, and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # DEBUG images shape
    # train_generator is DirectoryIterator yielding tuples of (x, y) where x is a
    # numpy array containing a batch of images with shape
    # (batch_size, *target_size, channels) and y is a numpy array of corresponding labels

    sample_batch = next(train_generator)
    print('Train img shape: ' + str(sample_batch[0].shape))

    # todo: reactivate this
    # this is a similar generator, for validation data
    # validation_generator = test_datagen.flow_from_directory(
    #        TRAIN_PATH,
    #        target_size=(150, 150),
    #        batch_size=BATCH_SIZE,
    #        class_mode='categorical')

    # TRAIN

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // BATCH_SIZE,
            epochs=50,
            validation_data=None,
            validation_steps=50 // BATCH_SIZE)

    # saving of model
    model_filename = 'model' + str(datetime.datetime.now().isoformat())

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename + '.json', "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_filename + '.h5')  # always save your weights after training or during training
    print("Saved model to disk as " + model_filename)
