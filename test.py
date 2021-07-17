from mnist_loader import load_data_wrapper
from MNIST_NN import NeuralNetwork

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)

net = NeuralNetwork(sizes=[784, 200, 10])
net.SGD(training_data=training_data,
        epochs_num=30,
        mini_batch_size=10,
        eta=3.0,
        test_data=test_data)