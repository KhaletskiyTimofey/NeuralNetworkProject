import InitFuncs
import LossFuncs
import ActivFuncs
import numpy as np

class LinearLayer:
    loss_name = None
    loss_func = None
    learning_rate = 0

    def __init__(self, input_size, output_size, activation_name, is_first_layer=False, is_last_layer=False):
        self.weights = np.float32(self.get_init_func(activation_name).init_weights(input_size, output_size))
        self.biases = np.zeros(output_size, "float32")

        self.activation_name = activation_name
        self.activation_func = self.get_activation_func(activation_name)
        self.is_need_cal_activ_deriv = self.activation_name != "linear"
        if is_last_layer:
            self.loss_func = self.get_loss_func(self.loss_name)
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.input_data = None
        self.output_data = None
        self.is_classification = None

    @staticmethod
    def get_init_func(activation_name):
        if activation_name in ("linear", "sigmoid", "tanh", "softmax"):
            return InitFuncs.Xavier()
        elif activation_name in ("relu", "leaky relu", "prelu"):
            return InitFuncs.He()
        else:
            raise NameError("Incorrect activation function name!")

    @staticmethod
    def get_activation_func(activation_name):
        return {
            "linear": ActivFuncs.Linear(),
            "sigmoid": ActivFuncs.Sigmoid(),
            "tanh": ActivFuncs.Tanh(),
            "softmax": ActivFuncs.Softmax(),
            "relu": ActivFuncs.ReLU(),
            "leaky relu": ActivFuncs.LeakyReLU()
        }.get(activation_name)

    def get_loss_func(self, loss_name):
        if loss_name == "mse":
            return LossFuncs.MSE()
        elif loss_name == "binary cross-entropy":
            if self.activation_name == "sigmoid":
                self.is_need_cal_activ_deriv = False
                return LossFuncs.BinaryCrossEntropy()
            else:
                raise NameError("Binary Cross-Entropy cannot be used with non-Sigmoid activation function!")
        elif loss_name == "categorical cross-entropy":
            if self.activation_name == "softmax":
                self.is_need_cal_activ_deriv = False
                return LossFuncs.CategoricalCrossEntropy()
            else:
                raise NameError("Categorical Cross-Entropy cannot be used with non-Softmax activation function!")
        else:
            raise NameError("Incorrect loss function name!")

    def forward_pass(self, input_data):
        self.input_data = np.float32(input_data)
        result = self.input_data @ self.weights + self.biases
        self.output_data = self.activation_func.calc(result)

        return self.output_data

    def backward_pass(self, desired_output):
        desired_output = np.float32(desired_output)
        batches_count = len(desired_output)

        if self.is_last_layer:
            delta = self.loss_func.calc_deriv(self.output_data, desired_output)
        else:
            delta = desired_output

        if self.is_need_cal_activ_deriv:
            delta *= self.activation_func.calc_deriv()

        l2_lambda = 0.001
        weights_grad = self.input_data.T @ delta + l2_lambda * self.weights
        biases_grad = np.sum(delta, axis=0)

        self.weights -= weights_grad * self.learning_rate
        self.biases -= biases_grad * self.learning_rate

        if not self.is_first_layer:
            return delta @ self.weights.T
        return None

    def get_loss(self):
        return self.loss_func.calc()