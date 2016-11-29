import sys

sys.path.append('..')
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from flickr import load_val_data, save_prediction
from keras.utils import np_utils
from util import Load, Save
from keras.callbacks import ModelCheckpoint, EarlyStopping

target_size = (224, 224)
train_size = 7000

batch_size = 32
nb_classes = 8
nb_epoch = 40
data_augmentation = False


def extract_vgg16():
    model = VGG16(weights='imagenet', include_top=False)
    print(model.summary())

    X_dirname = '../../411a3/train'
    Y_filename = '../../411a3/train.csv'
    X_filelist = image.list_pictures(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_vgg = np.zeros((train_size, 512, 7, 7))
    y_vgg = Y_list[:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(train_size):
        img = image.load_img(X_filelist[i], target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vgg16 = model.predict(x)
        X_vgg[i, :, :, :] = vgg16
        print('Read image: ' + X_filelist[i])

    # shuffle inputs and targets
    rnd_idx = np.arange(X_vgg.shape[0])
    np.random.shuffle(rnd_idx)
    X_train = X_vgg[rnd_idx]
    y_train = y_vgg[rnd_idx]

    return X_train, y_train


def save_vgg16(save_path='../../411a3/train_vgg16.npz'):
    X_train, y_train = extract_vgg16()
    train = {
        'X_train': X_train,
        'y_train': y_train
    }
    Save(save_path, train)
    print('Train data saved to: ' + save_path)


def load_vgg16(nb_test=0, load_path='../../411a3/train_vgg16.npz'):
    train = Load(load_path)
    X_train = train['X_train']
    y_train = train['y_train']

    # shuffle inputs and targets
    rnd_idx = np.arange(X_train.shape[0])
    np.random.shuffle(rnd_idx)
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]

    return (X_train[nb_test:], y_train[nb_test:]), (X_train[:nb_test], y_train[:nb_test])


def main():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = load_vgg16()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Flatten(name='flatten', input_shape=X_train.shape[1:]))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(8, activation='softmax', name='predictions'))

    print(model.summary())

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_split=0.1,
                  shuffle=True,
                  callbacks=[checkpoint, early_stopping])
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            callbacks=[checkpoint, early_stopping])

    X_val = load_val_data()
    prediction = model.predict_classes(X_val)
    save_prediction(prediction)


if __name__ == '__main__':
    main()
