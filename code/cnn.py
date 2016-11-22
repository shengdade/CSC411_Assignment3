from __future__ import division
from __future__ import print_function
from conv2d import conv2d as Conv2D
from nn import Affine, ReLU, AffineBackward, ReLUBackward, CheckGrad, Softmax, Train
import numpy as np
from nn import save_figure
from flickr import load_val_gist, save_prediction


def InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
            num_outputs):
    """Initializes CNN parameters.

    Args:
        num_channels:  Number of input channels.
        filter_size:   Filter size.
        num_filters_1: Number of filters for the first convolutional layer.
        num_filters_2: Number of filters for the second convolutional layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    W1 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_channels, num_filters_1)
    W2 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_filters_1, num_filters_2)
    W3 = 0.01 * np.random.randn(num_filters_2 * 64, num_outputs)
    b1 = np.zeros((num_filters_1))
    b2 = np.zeros((num_filters_2))
    b3 = np.zeros((num_outputs))
    v_W1 = np.zeros(W1.shape)
    v_W2 = np.zeros(W2.shape)
    v_W3 = np.zeros(W3.shape)
    v_b1 = np.zeros(b1.shape)
    v_b2 = np.zeros(b2.shape)
    v_b3 = np.zeros(b3.shape)
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'v_W1': v_W1,
        'v_W2': v_W2,
        'v_W3': v_W3,
        'v_b1': v_b1,
        'v_b2': v_b2,
        'v_b3': v_b3
    }
    return model


def MaxPool(x, ratio):
    """Computes non-overlapping max-pooling layer.

    Args:
        x:     Input values.
        ratio: Pooling ratio.

    Returns:
        y:     Output values.
    """
    xs = x.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio, int(xs[2] / ratio), ratio, xs[3]])
    y = np.max(np.max(h, axis=4), axis=2)
    return y


def MaxPoolBackward(grad_y, x, y, ratio):
    """Computes gradients of the max-pooling layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
    """
    dy = grad_y
    xs = x.shape
    ys = y.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio, int(xs[2] / ratio), ratio, xs[3]])
    y_ = np.expand_dims(np.expand_dims(y, 2), 4)
    dy_ = np.expand_dims(np.expand_dims(dy, 2), 4)
    dy_ = np.tile(dy_, [1, 1, ratio, 1, ratio, 1])
    dx = dy_ * (y_ == h).astype('float')
    dx = dx.reshape([ys[0], ys[1] * ratio, ys[2] * ratio, ys[3]])
    return dx


def Conv2DBackward(grad_y, x, y, w):
    """Computes gradients of the convolutional layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
        grad_w: Gradients wrt. the weights.
    """
    # Compute the padding
    pad = (w.shape[0] - 1, w.shape[1] - 1)

    # Update grad_w
    x_t = np.transpose(x, [3, 1, 2, 0])
    grad_y_t = np.transpose(grad_y, [1, 2, 0, 3])
    grad_w = Conv2D(x_t, grad_y_t, pad)
    grad_w = np.transpose(grad_w, [1, 2, 0, 3])

    # Update grad_x
    w_t = w[::-1, ::-1, :, :]
    w_t = np.transpose(w_t, [0, 1, 3, 2])
    grad_x = Conv2D(grad_y, w_t, pad)
    '''
    # Another way of updating grad_x
    I, J, C, K = w.shape
    w_t = np.zeros((I, J, K, C))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for c in range(C):
                    w_t[i, j, k, c] = w[I - i - 1, J - j - 1, c, k]
    grad_x = Conv2D(grad_y, w_t, pad)
    '''

    return grad_x, grad_w


def CNNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    x = x.reshape([-1, 32, 32, 1])
    h1c = Conv2D(x, model['W1']) + model['b1']
    h1r = ReLU(h1c)
    h1p = MaxPool(h1r, 2)
    h2c = Conv2D(h1p, model['W2']) + model['b2']
    h2r = ReLU(h2c)
    h2p = MaxPool(h2r, 2)
    h2p_ = np.reshape(h2p, [x.shape[0], -1])
    y = Affine(h2p_, model['W3'], model['b3'])
    var = {
        'x': x,
        'h1c': h1c,
        'h1r': h1r,
        'h1p': h1p,
        'h2c': h2c,
        'h2r': h2r,
        'h2p': h2p,
        'h2p_': h2p_,
        'y': y
    }
    return var


def CNNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2p_, dE_dW3, dE_db3 = AffineBackward(err, var['h2p_'], model['W3'])
    dE_dh2p = np.reshape(dE_dh2p_, var['h2p'].shape)
    dE_dh2r = MaxPoolBackward(dE_dh2p, var['h2r'], var['h2p'], 2)
    dE_dh2c = ReLUBackward(dE_dh2r, var['h2c'], var['h2r'])
    dE_dh1p, dE_dW2 = Conv2DBackward(
        dE_dh2c, var['h1p'], var['h2c'], model['W2'])
    dE_db2 = dE_dh2c.sum(axis=2).sum(axis=1).sum(axis=0)
    dE_dh1r = MaxPoolBackward(dE_dh1p, var['h1r'], var['h1p'], 2)
    dE_dh1c = ReLUBackward(dE_dh1r, var['h1c'], var['h1r'])
    _, dE_dW1 = Conv2DBackward(dE_dh1c, var['x'], var['h1c'], model['W1'])
    dE_db1 = dE_dh1c.sum(axis=2).sum(axis=1).sum(axis=0)
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass


def CNNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    # Update the weights.
    model['v_W1'] = momentum * model['v_W1'] + eps * model['dE_dW1']
    model['v_W2'] = momentum * model['v_W2'] + eps * model['dE_dW2']
    model['v_W3'] = momentum * model['v_W3'] + eps * model['dE_dW3']
    model['v_b1'] = momentum * model['v_b1'] + eps * model['dE_db1']
    model['v_b2'] = momentum * model['v_b2'] + eps * model['dE_db2']
    model['v_b3'] = momentum * model['v_b3'] + eps * model['dE_db3']

    model['W1'] = model['W1'] - model['v_W1']
    model['W2'] = model['W2'] - model['v_W2']
    model['W3'] = model['W3'] - model['v_W3']
    model['b1'] = model['b1'] - model['v_b1']
    model['b2'] = model['b2'] - model['v_b2']
    model['b3'] = model['b3'] - model['v_b3']


def main():
    """Trains a CNN."""
    model_fname = 'cnn_model.npz'
    stats_fname = 'cnn_stats.npz'

    # Hyper-parameters. Modify them if needed.
    eps = 0.1
    momentum = 0.5
    num_epochs = 30
    filter_size = 3
    num_filters_1 = 8
    num_filters_2 = 16
    batch_size = 100

    # Input-output dimensions.
    num_channels = 1
    num_outputs = 8

    # Initialize model.
    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                    num_outputs)

    # Uncomment to reload trained model here.
    # model = Load(model_fname)

    # Check gradient implementation.
    print('Checking gradients...')
    x = np.random.rand(10, 32, 32, 1) * 0.1
    CheckGrad(model, CNNForward, CNNBackward, 'W3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W1', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b1', x)

    # Train model.
    model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps, momentum, num_epochs, batch_size)

    train_ce_list = stats['train_ce']
    valid_ce_list = stats['valid_ce']
    train_acc_list = stats['train_acc']
    valid_acc_list = stats['valid_acc']

    save_figure(train_ce_list, valid_ce_list, 'ce', 'result/cnn')
    save_figure(train_acc_list, valid_acc_list, 'acc', 'result/cnn')

    # Uncomment if you wish to save the model.
    # Save(model_fname, model)

    # Uncomment if you wish to save the training statistics.
    # Save(stats_fname, stats)

    x = load_val_gist()
    var = CNNForward(model, x)
    prediction = np.argmax(Softmax(var['y']), axis=1)
    save_prediction(prediction)


if __name__ == '__main__':
    main()
