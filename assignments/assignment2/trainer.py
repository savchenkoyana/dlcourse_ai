from copy import deepcopy

import numpy as np
from metrics import multiclass_accuracy


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-2,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        """
        Computes accuracy on provided data using mini-batches
        """
        indices = np.arange(X.shape[0])
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]
        num_val = self.dataset.val_X.shape[0]
        
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for epoch in range(self.num_epochs):
            #first let's simply compute loss for validation data
            shuffled_indices = np.arange(num_val)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_val, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []

            for batch_indices in batches_indices:
                batch_X = self.dataset.val_X[batch_indices]
                batch_y = self.dataset.val_y[batch_indices]
                loss = self.model.compute_loss_and_gradients(batch_X, batch_y)

                # we're not updating weights here - we're not training, just computing loss
                
                batch_losses.append(loss)

            ave_loss = np.mean(batch_losses)            
            accuracy = self.compute_accuracy(self.dataset.val_X, self.dataset.val_y)
                                            
            val_loss_history.append(ave_loss)
            val_acc_history.append(accuracy)

            # now we're learning on train data 
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []

            for batch_indices in batches_indices:
                batch_X = self.dataset.train_X[batch_indices]
                batch_y = self.dataset.train_y[batch_indices]
                loss = self.model.compute_loss_and_gradients(batch_X, batch_y)
                
                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)

                batch_losses.append(loss)

            ave_loss = np.mean(batch_losses)
            
            accuracy = self.compute_accuracy(self.dataset.train_X, self.dataset.train_y)
                                            
            train_loss_history.append(ave_loss)
            train_acc_history.append(accuracy)


            if np.not_equal(self.learning_rate_decay, 1.0):
                self.learning_rate *= self.learning_rate_decay

            train_accuracy = train_acc_history[-1]
            val_accuracy = val_acc_history[-1]
            train_loss = train_loss_history[-1]
            val_loss = val_loss_history[-1]

            print("Train loss: %f, val loss: %f, train accuracy: %f, val accuracy: %f" %
                  (train_loss, val_loss, train_accuracy, val_accuracy))

        return train_loss_history, val_loss_history, train_acc_history, val_acc_history
