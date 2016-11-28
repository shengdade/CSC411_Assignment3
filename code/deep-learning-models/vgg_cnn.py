from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np

target_size = (224, 224)
train_size = 7000


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


def Save(fname, data):
    """Saves data to a numpy file."""
    print('Writing to ' + fname)
    np.savez_compressed(fname, **data)


def save_vgg16(save_path='../../411a3/train_vgg16.npz'):
    X_train, y_train = extract_vgg16()
    train = {
        'X_train': X_train,
        'y_train': y_train
    }
    Save(save_path, train)
    print('Train data saved to: ' + save_path)


if __name__ == '__main__':
    save_vgg16()
