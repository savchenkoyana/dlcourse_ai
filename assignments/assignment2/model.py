import numpy as np

from layers import Param, FullyConnectedLayer, ReLULayer, softmax, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.layers = [self.fc1, self.relu, self.fc2]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        x = X.copy()
        
        for name, param in self.params().items():
            param.grad = 0
        for layer in self.layers:
            x = layer.forward(x)
        
        loss, dprediction = softmax_with_cross_entropy(x, y)
        
        for layer in reversed(self.layers):
            dprediction = layer.backward(dprediction)
            
        for name, param in self.params().items():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
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
            if layer.params():
                for name, param in layer.params().items():
                    result["param_"+str(n_params)] = param
                    n_params += 1
        return result