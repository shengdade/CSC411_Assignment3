import numpy as np
from keras import backend as K
from keras.preprocessing import image


def load_data():
    X_dirname = '/Users/dadesheng/Workspace/Assignment3/411a3/train'
    Y_filename = '/Users/dadesheng/Workspace/Assignment3/411a3/train.csv'
    X_filelist = image.list_pictures(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_train = np.zeros((6500, 3, 32, 32))
    X_test = np.zeros((500, 3, 32, 32))
    y_train = Y_list[:6500, 1].astype('int64').reshape(-1, 1) - 1
    y_test = Y_list[6500:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(6500):
        img = image.load_img(X_filelist[i], target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_train[i, :, :, :] = x
        print('Processed image: ' + X_filelist[i])

    for i in range(6500, 7000):
        img = image.load_img(X_filelist[i], target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_test[i - 6500, :, :, :] = x
        print('Processed image: ' + X_filelist[i])

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test)


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x
