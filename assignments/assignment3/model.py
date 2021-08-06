import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        height, width, n_channels = input_shape
        
        pooling = 4
        stride = 4
        padding = 0
        kernel_size = 3
        
        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, kernel_size, padding)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pooling, stride)

        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, kernel_size, padding)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pooling, stride)
        self.flr = Flattener()
        
        size1 = ( (height-kernel_size+1-pooling)//stride + 1, (width-kernel_size+1-pooling)//stride + 1 )        
        size2 = ( (size1[0]-kernel_size+1-pooling)//stride + 1, (size1[1]-kernel_size+1-pooling)//stride + 1 )        

        self.fc = FullyConnectedLayer(size2[0]*size2[1]*conv2_channels, n_output_classes)
        
        self.layers = [self.conv1, self.relu1, self.maxpool1, 
                       self.conv2, self.relu2, self.maxpool2,
                       self.flr, self.fc]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
                
        for name, param in self.params().items():
            param.grad[:] = 0
            
        x = X.copy()

        for layer in self.layers:
            x = layer.forward(x)
        
        loss, dprediction = softmax_with_cross_entropy(x, y)

        for layer in reversed(self.layers):
            dprediction = layer.backward(dprediction)
        
        return loss
            
    def predict(self, X):
        x = X.copy()
        
        for layer in self.layers:
            x = layer.forward(x)
            
        probs = softmax(x)
        y_pred = np.argmax(probs, axis=1)
        return y_pred
        
    def params(self):
        result = {}
        n_params = 0
        for layer in self.layers:
            for name, param in layer.params().items():
                result['param'+str(n_params)] = param
                n_params += 1 
        return result
