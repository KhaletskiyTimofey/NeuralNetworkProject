import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
    def StandardScaler(input_data):
        data_mean = np.mean(input_data, axis=0)
        data_std = np.std(input_data, axis=0)
        return (input_data - data_mean) / (data_std + 1e-8)

    print("Preparing data...")

    dataset = pd.read_csv(r"Datasets\diabetes.csv")

    data = np.array(dataset.drop("Outcome", axis=1).values, "float32")
    results = np.array(dataset["Outcome"].values.reshape(-1, 1), "float32")

    for i in range(1, 6):
        parameter_mean = np.mean(data[data[:, i] != 0, i])
        data[data[:, i] == 0, i] = parameter_mean

    data_train, data_test, results_train, results_test = train_test_split(data, results,
                                                                          test_size=0.2,
                                                                          random_state=42,
                                                                          stratify=results)

    data_train = StandardScaler(data_train)

    print("Data prepared!")

    return data_train, data_test, results_train, results_test