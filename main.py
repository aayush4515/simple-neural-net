import numpy as np

# layers: each entry denotes the number of nodes in each layer
n = [2, 3, 3, 1]

def print_layer_sizes(n):
    print("layer 0 / input layer size", n[0])
    print("layer 1 size", n[1])
    print("layer 2 size", n[2])
    print("layer 3 size", n[3])

# randomly initialize weights and biases for each layer except input layer
# W -> Weight matrix, B -> Bias matrix

# dimension of a Weight matrix: Rows = # of nodes in the current layer, Columns = # of nodes in prev layer
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])

# dimension of a Bias matrix: Rows = # of nodes in the current layer, Colums = 1
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# function to check shapes
def check_shapes():
    print("Weights for layer 1 shape:", W1.shape)
    print("Weights for layer 2 shape:", W2.shape)
    print("Weights for layer 3 shape:", W3.shape)
    print("bias for layer 1 shape:", b1.shape)
    print("bias for layer 2 shape:", b2.shape)
    print("bias for layer 3 shape:", b3.shape)

# training data and labels
# m * n matrix (initially); m = number of samples, n = number of features
def prepare_data():
    # This data's features: n = 2; weight and height of individuals
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])

    # transpose the X matrix for feed forwarding, name it A0 (first layer / input layer)
    # dimension of A0: n[0] * m = 2 * 10
    A0 = X.T

    # y: training label data
    # 0: no risk for heart disease, 1: risk for heart disease
    y = np.array([
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0
    ])
    m = 10 # number of training inputs

    # need to reshape to a n[3] * m matrix for feed forwarding
    # Rows = number of output nodes for one training input, Cols = total # of inputs (m)
    Y = y.reshape(n[3], m)

    return A0, Y

# the activation function: sigmoid function
def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

# CALCULATIONS

def feed_forward(A0):
    # layer 1 calculations
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    # layer 2 calculations
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # layer 3 (output layer) calculations
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # y_hat: the prediction of the model
    y_hat = A3

    return y_hat

A0, Y = prepare_data()
y_hat = feed_forward(A0)

print(y_hat)