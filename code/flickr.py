import numpy as np
from keras import backend as K
from keras.preprocessing import image
import scipy.io
import glob
from util import Load

image_load_size = 224
gist_size = 1024


def load_image():
    X_dirname = '../411a3/train'
    Y_filename = '../411a3/train.csv'
    X_filelist = image.list_pictures(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_train = np.zeros((6500, 3, image_load_size, image_load_size))
    X_test = np.zeros((500, 3, image_load_size, image_load_size))
    y_train = Y_list[:6500, 1].astype('int64').reshape(-1, 1) - 1
    y_test = Y_list[6500:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(6500):
        img = image.load_img(X_filelist[i], target_size=(image_load_size, image_load_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_train[i, :, :, :] = x
        print('Read image: ' + X_filelist[i])

    for i in range(6500, 7000):
        img = image.load_img(X_filelist[i], target_size=(image_load_size, image_load_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_test[i - 6500, :, :, :] = x
        print('Read image: ' + X_filelist[i])

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test)


def load_val_image():
    X_dirname = '../411a3/val'
    X_filelist = image.list_pictures(X_dirname)
    val_samples = len(X_filelist)
    X_val = np.zeros((val_samples, 3, image_load_size, image_load_size))

    for i in range(val_samples):
        img = image.load_img(X_filelist[i], target_size=(image_load_size, image_load_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_val[i, :, :, :] = x
        print('Predict image: ' + X_filelist[i])

    if K.image_dim_ordering() == 'tf':
        X_val = X_val.transpose(0, 2, 3, 1)

    # print('X_val shape:', X_val.shape)

    return X_val


def load_gist():
    X_dirname = '../411a3/train_gist/*.mat'
    Y_filename = '../411a3/train.csv'
    X_filelist = glob.glob(X_dirname)
    Y_list = np.loadtxt(Y_filename, dtype='str', delimiter=',')[1:]

    X_train = np.zeros((6500, gist_size))
    X_test = np.zeros((500, gist_size))
    y_train = Y_list[:6500, 1].astype('int64').reshape(-1, 1) - 1
    y_test = Y_list[6500:, 1].astype('int64').reshape(-1, 1) - 1

    for i in range(6500):
        x = scipy.io.loadmat(X_filelist[i])['gist1']
        X_train[i, :] = x
        print('Read gist: ' + X_filelist[i])

    for i in range(6500, 7000):
        x = scipy.io.loadmat(X_filelist[i])['gist1']
        X_test[i - 6500, :] = x
        print('Read gist: ' + X_filelist[i])

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test)


def load_val_gist():
    X_dirname = '../411a3/val_gist/*.mat'
    X_filelist = glob.glob(X_dirname)
    val_samples = len(X_filelist)
    X_val = np.zeros((val_samples, gist_size))

    for i in range(val_samples):
        x = scipy.io.loadmat(X_filelist[i])['gist1']
        X_val[i, :] = x
        print('Predict gist: ' + X_filelist[i])

    # print('X_val shape:', X_val.shape)

    return X_val


def load_data():
    nb_test = 350
    train_file = '../411a3/train.npz'
    train = Load(train_file)
    X_train = train['X_train']
    y_train = train['y_train']
    rnd_idx = np.arange(X_train.shape[0])
    np.random.shuffle(rnd_idx)
    X_train = X_train[rnd_idx]
    y_train = y_train[rnd_idx]
    X_train = X_train.reshape((-1, 1, 32, 30))
    return (X_train[nb_test:], y_train[nb_test:]), (X_train[:nb_test], y_train[:nb_test])


def load_val_data():
    val_file = '../411a3/val.npz'
    val = Load(val_file)
    X_val = val['X_val']
    X_val = X_val.reshape((-1, 1, 32, 30))
    return X_val


def save_prediction(prediction):
    out = 'result/submission.csv'
    nb_val = 2970
    prediction += 1  # Class labels start at 1
    prediction_column = np.append(prediction, np.zeros(nb_val - prediction.size)).reshape(-1, 1)
    id_column = np.arange(nb_val).reshape(-1, 1) + 1
    result = np.concatenate((id_column, prediction_column), axis=1)
    np.savetxt(out, result, fmt='%d', delimiter=',', header='Id,Prediction', comments='')
    print('Prediction file written: ' + out)


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
