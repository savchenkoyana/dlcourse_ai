import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength* np.sum(W*W)
    grad = reg_strength* 2*W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    max_predictions = np.max(predictions) if predictions.ndim == 1 else np.max(predictions, axis=1).reshape((-1,1)) 
    predictions_shifted = predictions - max_predictions
    exps = np.exp(predictions_shifted)
    return exps/np.sum(exps) if predictions.ndim == 1 else exps/np.sum(exps, axis=1).reshape((-1,1))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N,1) or (batch_size, N) -
        probabilities for every class
      target_index: either int or np.array of shape (batch_size, 1) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    p = np.zeros(probs.shape)
    if probs.ndim == 2:
        p[np.arange(probs.shape[0]), target_index.flatten()] = 1
    elif probs.ndim == 1:
        p[target_index] = 1
        
    # in order to np.log not to crash with too small arguments
    probs[probs < 1.0e-100] = 1.0e-100
    
    return -np.sum(p * np.log(probs)) if probs.ndim == 1 else -np.sum(p * np.log(probs), axis=1).mean()
    

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    sm = softmax(predictions)
    loss = cross_entropy_loss(sm, target_index)
    p = np.zeros(sm.shape)
    batch_size = 1 if predictions.ndim == 1 else predictions.shape[0]
    
    if predictions.ndim == 1:
        p[target_index] = 1
    elif predictions.ndim == 2:
        p[np.arange(sm.shape[0]), target_index.flatten()] = 1

    dprediction = (sm-p)/batch_size
        
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X        
        return X*(X>0).astype(int)
            
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out*(self.X>0).astype(int)
        
    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value
        
    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        batch_size = d_out.shape[0]
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1, batch_size)), d_out)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2*self.padding + 1
        out_width = width - self.filter_size + 2*self.padding + 1
        
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        x = np.pad(X, pad_width=[(0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)])
        
        self.X = x
        
        for i in range(out_height):
            for j in range(out_width):
                x_flat = x[:, i:i+self.filter_size, j:j+self.filter_size, :]
                x_flat = x_flat.reshape((batch_size, -1))
                W_flat = self.W.value.reshape((-1, self.out_channels))
                output[:, i, j, :] = np.dot(x_flat, W_flat) + self.B.value
        return output

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_in = np.zeros(self.X.shape)
        W_flat = self.W.value.reshape((-1, self.out_channels)) 

        self.B.grad[:] = 0
        self.W.grad[:] = 0
        
        for i in range(out_height):
            for j in range(out_width):
                x_flat = self.X[:, i:i+self.filter_size, j:j+self.filter_size, :]
                x_flat = x_flat.reshape((batch_size, -1))
                
                d_in[:, i:i+self.filter_size, j:j+self.filter_size, :] += (
                    np.dot(d_out[:, i, j, :], W_flat.T)).reshape(self.X.shape)
                
                self.W.grad += np.dot(x_flat.T, d_out[:, i, j, :]).reshape(self.W.grad.shape)
                
                self.B.grad += np.dot(np.ones((1, batch_size)), d_out[:, i, j, :]).reshape(self.B.grad.shape)

        return d_in[:, self.padding : height-self.padding, self.padding : width-self.padding, :]
    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
