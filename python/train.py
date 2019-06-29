import datetime

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import backend as K
# TODO: investigate ordering in dimensions (to work now it set as Theano,
# channel first: (3,150,150))
K.set_image_dim_ordering('th')

INPUT_SHAPE = 150

def create_model():
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
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

if __name__ == '__main__':

    model = create_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    batch_size = 4

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of path, and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            '../images/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # DEBUG images shape
    # train_generator is DirectoryIterator yielding tuples of (x, y) where x is a
    # numpy array containing a batch of images with shape
    # (batch_size, *target_size, channels) and y is a numpy array of corresponding labels

    sample_batch = next(train_generator)
    print('Train img shape: ' + str((sample_batch[0])[0].shape))

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            '../images/train',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')

    # TRAIN

    model.fit_generator(
            train_generator,
            steps_per_epoch=10 // batch_size,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=50 // batch_size)

    # saving of model
    model_filename = 'model' + str(datetime.datetime.now().isoformat())

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_filename + '.json', "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_filename + '.h5')  # always save your weights after training or during training
    print("Saved model to disk as " + model_filename)
