from Dataset import get_data
from MultilayerPerceptron import *

data_train, data_test, results_train, results_test = get_data()

model = MultilayerPerceptron(input_size=8,
                             layers=[
                                 [8, "leaky relu"],
                                 [1, "sigmoid"]
                             ],
                             loss_name="binary cross-entropy",
                             learning_rate=0.001)

model.train(input_data=data_train,
            desired_output=results_train,
            epochs=1000,
            is_full_batch=True,
            batch_size=32)

model.get_accuracy(data_train, results_train, accuracy=0.1)
model.get_accuracy(data_test, results_test, accuracy=0.1)
model.get_loss_graph()

while True:
    pregnancies = float(input("Input pregnancies count: "))
    glucose = float(input("Input glucose: "))
    blood_pressure = float(input("Input blood pressure: "))
    skin_thickness = float(input("Input skin thickness: "))
    insulin = float(input("Input insulin: "))
    BMI = float(input("Input BMI: "))
    diabetes_pedigree_function = float(input("Input diabetes pedigree function: "))
    age = float(input("Input age: "))

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        BMI,
        diabetes_pedigree_function,
        age
    ]], "float32")

    output = model.forward_pass(data)[0, 0]

    print(f"\nOutput: {round(float(output), 2)}")