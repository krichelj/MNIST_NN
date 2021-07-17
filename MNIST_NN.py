import random
import numpy as np
from typing import List, Tuple


class NeuralNetwork:
    """
    Neural network base class

    Parameters
    ----------
    sizes:
        A list of the number of neurons in each layer.
        For a network with 2 neurons in the first layer,
        3 neurons in the second layer, and 1 neuron in the final layer:
        list = [2, 3, 1]
    """

    def __init__(self, sizes: List[int]):
        self.L = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

    def feedforward(self, x: np.array) -> np.array:
        """
        Applies the feedforward rule for all L layers at once

        Parameters
        ----------
        x:
            The vector of values at the input layer

        Returns
        -------
        x:
            The vector of activations at the output layer
        """
        for W_l, b_l in zip(self.weights, self.biases):
            x = sigmoid(W_l @ x + b_l)

        return x

    def SGD(self, training_data: List[Tuple[np.array, np.array]], epochs_num: int,
            mini_batch_size: int, eta: float, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent

        Parameters
        ----------
        training_data:
            A list of tuples (x, y) representing
            the training examples and the desired outputs
        epochs_num:
            Number of epochs to run
        mini_batch_size:
            The size of one mini-batch
        eta:
            The learning rate
        test_data:
            The test set. If provided then the network
            will be evaluated against the test data after each
            epoch, and partial progress printed out. This is useful for
            tracking progress, but slows things down substantially
        """

        m = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for epoch in range(1, epochs_num + 1):
            random.shuffle(training_data)

            for i in range(0, m, mini_batch_size):
                current_mini_batch = training_data[i:i + mini_batch_size]
                self.update_mini_batch(mini_batch=current_mini_batch,
                                       eta=eta)
            if test_data:
                print(f"Epoch {epoch} complete, with: "
                      f"{int(self.evaluate(test_data) * 100/ n_test)}% accuracy on the test set")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch: List[Tuple[np.array, np.array]], eta: float):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch

        Parameters
        ----------
        mini_batch:
            List of tuples (x, y) of examples and labels
        eta:
            The learning rate
        """

        m = len(mini_batch)
        zeta = eta / m

        for x, y in mini_batch:
            nabla_W, nabla_b = self.backprop(x, y)

            for i, (dC_dW_i, dC_db_i) in enumerate(zip(nabla_W, nabla_b)):
                self.weights[i] -= zeta * dC_dW_i
                self.biases[i] -= zeta * dC_db_i

    def backprop(self, x: np.array, y: np.array) -> Tuple[List[np.array], List[np.array]]:
        """
        Implement backpropagation
        Return a tuple representing the gradients for the cost function C_x

        C(W_i=1^n,b_i=1^n) = 1/2 sum_x=1^n (y(x) - a_L)^2
        z_l = W_l * a_(l-1) + b_l
        a_l = sigma(z_l)


        Returns
        -------
        dC/dW:
            Layer-by-layer numpy arrays representing the derivatives
            of the cost function with respect to the weights
        dC/dB:
            Layer-by-layer numpy arrays representing the derivatives
            of the cost function with respect to the biases

        Notes
        -------

        Using the chain rule, we have:
            dC/dW_l = dC/da_l * da_l/dz_l * dz_l/dW_l
            dC/db_l = dC/da_l * da_l/dz_l * dz_l/db_l

            Where for the output layer:
            dC/da_L = a_L - y(x)

            And from there backwards:
            dC/da_(l-1) = dC/da_l * da_l/dz_l * dz_l/da_(l-1)
            dz_l/da_(l-1) = W_l
            da_l/dz_l = sigmoid_prime(z_l)

            Moreover:
            dz_l/dW_l = a_(l-1)
            dz_l/db_l = 1

            And thus:
            ----> For the output layer:
            ----> dC/dW_(L-1) = (a_L - y(x)) * sigmoid_prime(z_L) * a_(L-1)
            ----> dC/db_(L-1) = (a_L - y(x)) * sigmoid_prime(z_L)

            ----> For 2 <= l <= L -1:
            ----> dC/dW_(l-1) = dC/da_(l-1) * da_(l-1)/dz_(l-1) * dz_(l-1)/dW_(l-1)
                              = dC/da_l * da_l/dz_l * dz_l/da_(l-1) * da_(l-1)/dz_(l-1) * dz_(l-1)/dW_(l-1)
                              = dC/da_l * sigmoid_prime(z_l) * W_l * sigmoid_prime(z_(l-1)) * a_(l-1)
        """

        nabla_W = [np.zeros(W_l.shape) for W_l in self.weights]
        nabla_b = [np.zeros(b_l.shape) for b_l in self.biases]

        A = [x]
        Z = []

        for W_l, b_l in zip(self.weights, self.biases):
            z_l = W_l @ A[-1] + b_l
            Z += [z_l]
            a_l = sigmoid(z_l)
            A += [a_l]

        a_L = A[-1]
        z_L = Z[-1]
        dC_da_L = self.cost_derivative(a_L, y)
        da_L_dz_L = sigmoid_prime(z_L)
        dC_dz_L = dC_da_L * da_L_dz_L
        a_L_1 = A[-2]
        dz_L_dW_L = a_L_1
        dC_dW_L = dC_dz_L @ dz_L_dW_L.transpose()
        dC_db_L = dC_dz_L

        nabla_W[-1] = dC_dW_L
        nabla_b[-1] = dC_db_L

        dC_dz_l = dC_dz_L

        for l_1 in range(2, self.L):
            W_l = self.weights[-l_1 + 1]
            dz_l_da_l_1 = W_l
            z_l_1 = Z[-l_1]
            sp_z_l_1 = sigmoid_prime(z_l_1)
            da_l_1_dz_l_1 = sp_z_l_1
            a_l_1 = A[-l_1 - 1]
            dz_l_1_dW_l_1 = a_l_1

            dC_da_l_1 = dC_dz_l.transpose() @ dz_l_da_l_1
            dC_dz_l_1 = dC_da_l_1.transpose() * da_l_1_dz_l_1
            dC_dW_l_1 = dC_dz_l_1 @ dz_l_1_dW_l_1.transpose()
            dC_db_l_1 = dC_dz_l_1

            nabla_W[-l_1] = dC_dW_l_1
            nabla_b[-l_1] = dC_db_l_1

            # here comes the update:
            dC_dz_l = dC_dz_l_1

        return nabla_W, nabla_b

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation
        """
        return sum(int(np.argmax(self.feedforward(x)) == y) for (x, y) in test_data)

    def cost_derivative(self, a_output, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return a_output - y


def sigmoid(x: np.array) -> np.array:
    """The sigmoid function"""
    return 1 / (1.0 + np.exp(-x))


def sigmoid_prime(x: np.array) -> np.array:
    """The derivative of the sigmoid function"""
    sigmoid_x = sigmoid(x)

    return sigmoid_x * (1 - sigmoid_x)
