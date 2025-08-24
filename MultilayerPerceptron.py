import time
import random
from LinearLayer import *
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self, input_size, layers, loss_name, learning_rate):
        LinearLayer.loss_name = loss_name
        LinearLayer.learning_rate = learning_rate

        self.linear_layers = []
        last_output_size = input_size

        for i in range(len(layers)):
            layer = layers[i]
            self.linear_layers.append(LinearLayer(last_output_size, layer[0], layer[1],
                                                  is_first_layer=i == 0,
                                                  is_last_layer=i == len(layers) - 1))
            last_output_size = layer[0]

        self.linear_layers = tuple(self.linear_layers)

        self.epochs_idx = None
        self.loss_values = None

    def forward_pass(self, input_data):
        output_data = input_data
        for layer in self.linear_layers:
            output_data = layer.forward_pass(output_data)

        return output_data

    def backward_pass(self, desired_output):
        prev_layer_grad = desired_output
        for layer in reversed(self.linear_layers):
            prev_layer_grad = layer.backward_pass(prev_layer_grad)

    def train(self, input_data, desired_output, epochs, is_full_batch=True, batch_size=0):
        self.epochs_idx = []
        self.loss_values = []

        print("Training started...")
        start_time = time.perf_counter()

        for epoch in range(epochs):
            if not is_full_batch:
                start_idx = random.randint(0, len(input_data) - batch_size)

                input_batches = input_data[start_idx:start_idx + batch_size]
                desired_output_batches = desired_output[start_idx:start_idx + batch_size]
            else:
                input_batches = input_data
                desired_output_batches = desired_output

            self.forward_pass(input_batches)
            self.backward_pass(desired_output_batches)

            self.epochs_idx.append(epoch)
            self.loss_values.append(self.linear_layers[-1].get_loss())

        print("Training completed!")
        print(f"Time spent: {round(time.perf_counter() - start_time, 3)} s")

    def get_loss_graph(self):
        plt.plot(self.epochs_idx, self.loss_values)
        plt.show()

    def get_accuracy(self, data_test, results_test, accuracy=0.1):
        output_results = self.forward_pass(data_test)

        results_difference = output_results - results_test

        is_min_incorrect = np.float32(np.abs(np.min(results_difference, axis=1)) > accuracy)
        is_max_incorrect = np.float32(np.abs(np.max(results_difference, axis=1)) > accuracy)

        correct_results = np.sum((is_min_incorrect == 0) & (is_max_incorrect == 0))

        print(f"Accuracy: {round(correct_results / len(data_test) * 100)}%")