"""
Instruction:

In this section, you are asked to train a CNN with different hyperparameters.
To start with training, you need to fill in the incomplete code. There are two
places that you need to complete:
a) Backward pass equations for a convolutional layer.
b) Weight update equations with momentum.

After correctly fill in the code, modify the hyperparameters in "main()".
You can then run this file with the command: "python cnn.py" in your terminal.
The program will automatically check your gradient implementation before start.
The program will print out the training progress, and it will display the
training curve by the end. You can optionally save the model by uncommenting
the lines in "main()". You can also optionally load a trained model by
uncommenting the lines before the training.

Important Notes:
1. The Conv2D function has already been implemented. To implement the backward
pass, you should only need to call the existing Conv2D function, following the
equations from part 2 of the assignment. An efficient solution should be within
10 lines of code.

2. The Conv2D function accepts an optional parameter "pad", by default it will
pad the input with the size of the filter, so that the output has the same
dimension as the input. For example,
-----------------------------------------------------------------------------
Variable                | Shape
-----------------------------------------------------------------------------
x (Inputs)              | [N, H, W, C]
w (Filters)             | [I, J, C, K]
Conv2D(x, w)            | [N, H+I, W+J, C] * [I, J, C, K] = [N, H, W, K]
-----------------------------------------------------------------------------
(N=number of examples, H=height, W=width, C=number of channels,
 I=filter height, J=filter width, K=number of filters)

You can also pass in a tuple (P, Q) to specify the amount you want to pad on
height and width dimension.
For example,
-----------------------------------------------------------------------------
Variable                | Shape
-----------------------------------------------------------------------------
x (Inputs)              | [N, H, W, C]
w (Filters)             | [I, J, C, K]
Conv2D(x, w, pad=(P,Q)) | [N, H+P, W+Q, C] * [I, J, C, K] = [N, H+P-I, W+Q-J, K]
-----------------------------------------------------------------------------

3. It is maybe helpful to use "np.transpose" function to transpose the
dimensions of a variable. For example:
-----------------------------------------------------------------------------
Variable                   | Shape
-----------------------------------------------------------------------------
x (Inputs)                 | [N, H, W, C]
np.transpose(x, [3,1,2,0]) | [C, H, W, N]
-----------------------------------------------------------------------------
"""

from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
from conv2d import conv2d as Conv2D
from nn import Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate, StartCheckGrad

import numpy as np


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
    W1 = 0.1 * np.random.randn(filter_size, filter_size, num_channels,  num_filters_1)
    W2 = 0.1 * np.random.randn(filter_size, filter_size, num_filters_1, num_filters_2)
    W3 = 0.01 * np.random.randn(num_filters_2 * 64, num_outputs)
    b1 = np.zeros((num_filters_1))
    b2 = np.zeros((num_filters_2))
    b3 = np.zeros((num_outputs))


    dE_dW1old = 0 * np.random.randn(filter_size, filter_size, num_channels,  num_filters_1)
    dE_dW2old = 0 * np.random.randn(filter_size, filter_size, num_filters_1, num_filters_2)
    dE_dW3old = 0 * np.random.randn(num_filters_2 * 64, num_outputs)

    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'dE_dW1old': dE_dW1old,
        'dE_dW2old': dE_dW2old,
        'dE_dW3old': dE_dW3old,
        'dE_db1old': 0,
        'dE_db2old': 0,
        'dE_db3old': 0
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
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
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
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
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
    ###########################
    I = w.shape[0] #height
    J = w.shape[1] #width
    P = I-1
    Q = J-1

    w = np.fliplr(w)
    w = np.flipud(w)
    w = np.transpose(w, [0,1,3,2])

    grad_x = Conv2D(grad_y, w, pad=(P,Q))
    x = np.transpose(x, [3, 1, 2, 0])
    grad_y = np.transpose(grad_y, [1, 2, 0, 3])
    grad_w = Conv2D(x, grad_y, pad=(P, Q))
    grad_w = np.transpose(grad_w, [1, 2, 0, 3])


    return grad_x, grad_w
    raise Exception('Not implemented')


def CNNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    x = x.reshape([-1, 48, 48, 1])
    h1c = Conv2D(x, model['W1']) + model['b1']
    h1r = ReLU(h1c)
    h1p = MaxPool(h1r, 3)
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
    dE_dh1r = MaxPoolBackward(dE_dh1p, var['h1r'], var['h1p'], 3)
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
    ###########################
    # Insert your code here.
    # Update the weights.

    model['dE_dW1old'] = - eps * model['dE_dW1'] + model['dE_dW1old'] * momentum
    model['dE_dW2old'] = - eps * model['dE_dW2'] + model['dE_dW2old'] * momentum
    model['dE_dW3old'] = - eps * model['dE_dW3'] + model['dE_dW3old'] * momentum
    model['dE_db1old'] = - eps * model['dE_db1'] + model['dE_db1old'] * momentum
    model['dE_db2old'] = - eps * model['dE_db2'] + model['dE_db2old'] * momentum
    model['dE_db3old'] = - eps * model['dE_db3'] + model['dE_db3old'] * momentum

    model['W1'] = model['W1'] + model['dE_dW1old']
    model['W2'] = model['W2'] + model['dE_dW2old']
    model['W3'] = model['W3'] + model['dE_dW3old']
    model['b1'] = model['b1'] + model['dE_db1old']
    model['b2'] = model['b2'] + model['dE_db2old']
    model['b3'] = model['b3'] + model['dE_db3old']

    #raise Exception('Not implemented')


def main():
    """Trains a CNN."""
    
    runType = "CNN"
    
    model_fname = 'cnn_model.npz'
    stats_fname = 'cnn_stats.npz'

    # Hyper-parameters. Modify them if needed.
    filter_size = 5
    num_channels = 1
    num_outputs = 7
    num_filters_1 = 8
    num_filters_2 = 16
    hid1 = num_filters_1
    hid2 = num_filters_2


    loop = 0
    # loop = 0 for regular
    # loop = 1 for 75 plots
    # loop = 2 for 13 plots
    # loop = 3 for Q3.3
    # loop = 4 for Q3.4

    q34 = False

    if loop == 1: #for generating 75 plots

        eps = np.array([0.1, 0.001, 0.01, 0.5, 1])
        momentum = np.array([0.0, 0.45, 0.9])
        num_epochs = np.array([30])
        batch_size = np.array([100, 1000, 500, 10, 1])

        for i in xrange(eps.shape[0]):
            for j in xrange(momentum.shape[0]):
                for k in xrange(num_epochs.shape[0]):
                    for l in xrange(batch_size.shape[0]):
                        # Initialize model.

                        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,num_outputs)
                        StartCheckGrad(model, CNNForward, CNNBackward)
                        stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps[i],
                                      momentum[j], num_epochs[k], batch_size[l], hid1, hid2, q34, runType)

    if loop == 2: #for generating 13 plots

        eps = np.array([0.1, 0.001, 0.01, 0.5, 1])
        momentum = np.array([0.0, 0.45, 0.9])
        num_epochs = np.array([30])
        batch_size = np.array([100, 1000, 500, 10, 1])

        for i in xrange(eps.shape[0]):
            model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_outputs)
            StartCheckGrad(model, CNNForward, CNNBackward)
            stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps[i],
                            momentum[0], num_epochs[0], batch_size[0], hid1, hid2, q34, runType)
        for i in xrange(momentum.shape[0]):
            model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_outputs)
            StartCheckGrad(model, CNNForward, CNNBackward)
            stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps[0],
                            momentum[i], num_epochs[0], batch_size[0], hid1, hid2, q34, runType)
        for i in xrange(batch_size.shape[0]):
            model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_outputs)
            StartCheckGrad(model, CNNForward, CNNBackward)
            stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps[0],
                            momentum[0], num_epochs[0], batch_size[i], hid1, hid2, q34, runType)


    if loop == 3: #for generating Q3.3

        eps = np.array([0.01])
        momentum = np.array([0.9])
        num_epochs = np.array([30])
        batch_size = np.array([100])

        num_filters_1 = np.array([2, 16, 16, 32])
        num_filters_2 = np.array([2, 8, 32, 16])

        for i in range(num_filters_1.shape[0]):
            hid1 = num_filters_1[i]
            hid2 = num_filters_2[i]
            model = InitCNN(num_channels, filter_size, hid1, hid2, num_outputs)
            StartCheckGrad(model, CNNForward, CNNBackward)
            stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps[0], momentum[0], num_epochs[0], batch_size[0], hid1, hid2, q34, runType)

    if loop == 4: #for generating Q3.4
        q34 = True
        num_filters_1 = 8
        num_filters_2 = 16
        hid1 = num_filters_1
        hid2 = num_filters_2

        eps = 0.1
        momentum = 0.9
        num_epochs = 30
        batch_size = 100

        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        StartCheckGrad(model, CNNForward, CNNBackward)
        stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
                      momentum, num_epochs, batch_size, hid1, hid2, q34, runType)

    if loop == 0:  #loop = 0

        eps = 0.1
        momentum = 0.0
        num_epochs = 30
        batch_size = 100

        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        StartCheckGrad(model, CNNForward, CNNBackward)
        stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
                      momentum, num_epochs, batch_size, hid1, hid2, q34, runType)

if __name__ == '__main__':
    main()
