from Dataset import get_data
from MultilayerPerceptron import *

x_train, x_test, y_train, y_test = get_data()

model = MultilayerPerceptron(input_size=8,
                             layers=[
                                 [8, "relu"],
                                 [1, "sigmoid"]
                             ],
                             loss_name="binary cross-entropy",
                             learning_rate=0.00075)

model.train(input_data=x_train,
            desired_output=y_train,
            epochs=6000)

model.get_accuracy(x_train, y_train, accuracy=0.5)
model.get_accuracy(x_test, y_test, accuracy=0.5)
model.get_loss_graph()