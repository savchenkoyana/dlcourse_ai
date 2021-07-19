import numpy as np

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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength* np.sum(W*W)
    grad = reg_strength* 2*W
    
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, X_val=None, y_val=None):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []

        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            n_batches = len(batches_indices)
            loss = 0
            for indices in batches_indices:
                batch_loss, dW = linear_softmax(X[indices], self.W, y[indices])
                self.W -= learning_rate * dW
                
                loss += batch_loss
                
            loss /= n_batches
            l2_loss, l2_grad = l2_regularization(self.W, reg)
            loss += l2_loss
            self.W -= learning_rate * l2_grad
            
            loss_history.append(loss)
            	
            if epoch%10==0:
                if X_val is not None:
                    val_loss, _ = linear_softmax(X_val, self.W, y_val)
                    val_loss += l2_regularization(self.W, reg)[0]
                    
                    print("Epoch %i, loss: %f, val loss %f" %(epoch, loss, val_loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        pred = np.dot(X, self.W)
        sm = softmax(pred)
        y_pred = np.argmax(sm, axis=1)
        
        return y_pred
