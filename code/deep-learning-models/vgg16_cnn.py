from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

target_size = (224, 224)
train_size = 7000
val_size = 970
test_size = 2000

batch_size = 32
nb_classes = 8
nb_epoch = 40
data_augmentation = False

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


def Save(fname, data):
    """Saves data to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def Load(fname):
    """Loads data from numpy file."""
    print('Loading from ' + fname)
    return dict(np.load(fname))


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


def extract_vgg16_val():
    model = VGG16(weights='imagenet', include_top=False)
    print(model.summary())

    X_dirname = '../../411a3/val'
    X_filelist = image.list_pictures(X_dirname)

    X_vgg_val = np.zeros((val_size, 512, 7, 7))

    for i in range(val_size):
        img = image.load_img(X_filelist[i], target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vgg16 = model.predict(x)
        X_vgg_val[i, :, :, :] = vgg16
        print('Read image: ' + X_filelist[i])

    return X_vgg_val


def extract_vgg16_test():
    model = VGG16(weights='imagenet', include_top=False)
    print(model.summary())

    X_dirname = '../../411a3/test'
    X_filelist = image.list_pictures(X_dirname)

    X_vgg_test = np.zeros((test_size, 512, 7, 7))

    for i in range(test_size):
        img = image.load_img(X_filelist[i], target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vgg16 = model.predict(x)
        X_vgg_test[i, :, :, :] = vgg16
        print('Read image: ' + X_filelist[i])

    return X_vgg_test


def save_vgg16(save_path='../../411a3/train_vgg16.npz'):
    X_train, y_train = extract_vgg16()
    train = {
        'X_train': X_train,
        'y_train': y_train
    }
    Save(save_path, train)
    print('Train data saved to: ' + save_path)


def save_vgg16_val(save_path='../../411a3/train_vgg16_val.npz'):
    X_val = extract_vgg16_val()
    val = {
        'X_val': X_val
    }
    Save(save_path, val)
    print('Val data saved to: ' + save_path)


def save_vgg16_test(save_path='../../411a3/train_vgg16_test.npz'):
    X_test = extract_vgg16_test()
    test = {
        'X_test': X_test
    }
    Save(save_path, test)
    print('Test data saved to: ' + save_path)


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


def load_vgg16_val(load_path='../../411a3/train_vgg16_val.npz'):
    val = Load(load_path)
    X_val = val['X_val']

    return X_val


def load_vgg16_test(load_path='../../411a3/train_vgg16_test.npz'):
    test = Load(load_path)
    X_test = test['X_test']

    return X_test


def save_prediction(val, test):
    out = '../result/submission.csv'
    nb_predict = 2970
    val += 1  # Class labels start at 1
    test += 1  # Class labels start at 1
    prediction_column = np.append(val, test).reshape(-1, 1)
    id_column = np.arange(nb_predict).reshape(-1, 1) + 1
    result = np.concatenate((id_column, prediction_column), axis=1)
    np.savetxt(out, result, fmt='%d', delimiter=',', header='Id,Prediction', comments='')
    print('Prediction file written: ' + out)


def create_model(input_shape):
    model = Sequential()
    model.add(Flatten(name='flatten', input_shape=input_shape))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(8, activation='softmax', name='predictions'))
    return model


def main():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = load_vgg16()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = create_model(X_train.shape[1:])
    print(model.summary())

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

    X_val = load_vgg16_val()
    X_test = load_vgg16_test()
    prediction_val = model.predict_classes(X_val)
    prediction_test = model.predict_classes(X_test)
    save_prediction(prediction_val, prediction_test)


def load_model_predict(weights_file):
    X_val = load_vgg16_val()
    X_test = load_vgg16_test()
    model = create_model(X_val.shape[1:])
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print("Created model and loaded weights from file")
    prediction_val = model.predict_classes(X_val)
    prediction_test = model.predict_classes(X_test)
    save_prediction(prediction_val, prediction_test)


if __name__ == '__main__':
    main()
