import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


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
        
        for param in self.params():
            param.grad = 0
        for layer in self.layers:
            x = layer.forward(x)
        
        loss, dprediction = softmax_with_cross_entropy(x, y)
        
        for layer in reversed(self.layers):
            dprediction = layer.backward(dprediction)
            
        for param in self.params():
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
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}
        for layer in self.layers:
            if layer.params():
                name, param = layer.params()
                result[name] = layer.params()
        return result
