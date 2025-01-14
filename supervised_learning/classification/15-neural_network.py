#!/usr/bin/env python3
"""Importing numpy"""


import numpy as np


class NeuralNetwork:
    """
    initializing the class
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        nx: no of input features
        nodes: no of nodes in hidden layer
        W1: weights for hidden layer
        b1: bias of hidden layer
        A1: activated output for hidden layer
        W2: weights of output
        b2: bias for output
        A2: activated output
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter function"""
        return self.__W1

    @property
    def b1(self):
        """Getter function b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter function A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter function W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter function for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter function A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        calculates forward prop
        X: np array shape(nx, m)
        nx: no of input features
        m: no of examples
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calculates cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate neural networks predictions
        """
        __, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dW2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        # Update weights and biases
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    #!/usr/bin/env python3
"""Importing numpy"""


import numpy as np


class NeuralNetwork:
    """
    initializing the class
    """
    def __init__(self, nx, nodes):
        """
        Class constructor
        nx: no of input features
        nodes: no of nodes in hidden layer
        W1: weights for hidden layer
        b1: bias of hidden layer
        A1: activated output for hidden layer
        W2: weights of output
        b2: bias for output
        A2: activated output
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter function"""
        return self.__W1

    @property
    def b1(self):
        """Getter function b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter function A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter function W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter function for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter function A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        calculates forward prop
        X: np array shape(nx, m)
        nx: no of input features
        m: no of examples
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calculates cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate neural networks predictions
        """
        __, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dW2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        # Update weights and biases
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Training the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0.0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        plotting_costs = []
        if graph:
            plotting_steps = np.arange(0, iterations + 1, step)
            if iterations % step != 0:
                plotting_steps = np.append(plotting_steps, iterations)

            for iteration in range(iterations):
                A1, A2 = self.forward_prop(X)
                self.gradient_descent(X, Y, A1, A2, alpha)
                if (iteration % step) == 0 or iteration == (iterations - 1):
                    cost = self.cost(Y, A2)
                    plotting_costs.append(cost)

                    if verbose:
                        print(f"Cost after {iteration} iterations: {cost}")

                # plotting
            plt.plot(plotting_steps, plotting_costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
