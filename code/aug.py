import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, list_pictures
from PIL import Image
from util import Save, Load
# In order to run the script on ECE hosts, the following is disabled.
# import leargist

image_size = 128
image_channel = 3
train_size = 7000
nb_classes = 8
gist_size = 960


def load_image(nb_test=0):
    X_dirname = '../411a3/train'
    Y_filename = '../411a3/train.csv'
    X_filelist = list_pictures(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_train = np.zeros((train_size, image_channel, image_size, image_size))
    y_train = Y_list[:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(train_size):
        img = load_img(X_filelist[i])
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        X_train[i, :, :, :] = x
        print('Read image: ' + X_filelist[i])

    # shuffle inputs and targets
    rnd_idx = np.arange(X_train.shape[0])
    np.random.shuffle(rnd_idx)
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)

    # print('X_train shape:', X_train.shape)
    # print('y_train shape:', y_train.shape)

    return (X_train[nb_test:], y_train[nb_test:]), (X_train[:nb_test], y_train[:nb_test])


def augment_image(train_upper, test_upper):
    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    (X_train, y_train), (X_test, y_test) = load_image(500)

    # augment train images to upper number each class
    for i_class in range(nb_classes):
        index = np.where(y_train == i_class)[0]
        x = X_train[index]
        # nb_org_image = x.shape[0]
        nb_aug_image = train_upper
        print('train class ' + str(i_class) + ' has ' + str(x.shape))
        i = 0
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir='../411a3/data/train/',
                                  save_format='jpg'):
            i += 1
            if i >= nb_aug_image:
                break  # otherwise the generator would loop indefinitely

    # augment validation images to upper number each class
    for i_class in range(nb_classes):
        index = np.where(y_test == i_class)[0]
        x = X_test[index]
        # nb_org_image = x.shape[0]
        nb_aug_image = test_upper
        print('validation class ' + str(i_class) + ' has ' + str(x.shape))
        i = 0
        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir='../411a3/data/validation/',
                                  save_format='jpg'):
            i += 1
            if i >= nb_aug_image:
                break  # otherwise the generator would loop indefinitely


def load_aug_image_train():
    X_file_base = '../411a3/data_classify/train/'
    X_train = np.empty((0, gist_size))
    y_train = np.empty((0, 1))
    for i_class in range(nb_classes):
        X_file_part = X_file_base + str(i_class)
        filelist = list_pictures(X_file_part)
        nb_part = len(filelist)
        X_part = np.zeros((nb_part, gist_size))
        for i in range(nb_part):
            im = Image.open(filelist[i])
            descriptors = leargist.color_gist(im)
            X_part[i, :] = descriptors
            print('Load image: ' + filelist[i])
        X_train = np.concatenate((X_train, X_part), axis=0)
        y_train = np.concatenate((y_train, np.ones((nb_part, 1)) * i_class), axis=0)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    return X_train, y_train


def load_aug_image_validation():
    X_file_base = '../411a3/data_classify/validation/'
    X_validation = np.empty((0, gist_size))
    y_validation = np.empty((0, 1))
    for i_class in range(nb_classes):
        X_file_part = X_file_base + str(i_class)
        filelist = list_pictures(X_file_part)
        nb_part = len(filelist)
        X_part = np.zeros((nb_part, gist_size))
        for i in range(nb_part):
            im = Image.open(filelist[i])
            descriptors = leargist.color_gist(im)
            X_part[i, :] = descriptors
            print('Load image: ' + filelist[i])
        X_validation = np.concatenate((X_validation, X_part), axis=0)
        y_validation = np.concatenate((y_validation, np.ones((nb_part, 1)) * i_class), axis=0)

    print('X_validation shape:', X_validation.shape)
    print('y_validation shape:', y_validation.shape)

    return X_validation, y_validation


def save_train():
    train_file = '../411a3/train_aug.npz'
    X_train, y_train = load_aug_image_train()
    X_test, y_test = load_aug_image_validation()
    train = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    Save(train_file, train)
    print('Train data saved to: ' + train_file)


def load_aug_data():
    train_file = '../411a3/train_aug.npz'
    train = Load(train_file)

    X_train = train['X_train']
    y_train = train['y_train']
    X_test = train['X_test']
    y_test = train['y_test']

    rnd_idx = np.arange(X_train.shape[0])
    np.random.shuffle(rnd_idx)
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]

    X_train = X_train.reshape((-1, 1, 32, 30))
    X_test = X_test.reshape((-1, 1, 32, 30))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    load_aug_data()
