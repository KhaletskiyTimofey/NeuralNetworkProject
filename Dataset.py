import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def StandardScaler(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + 1e-6)

def get_data():
    '''print("Preparing data...")

    data = pd.read_csv(r"Datasets\diabetes.csv")

    x = data.drop("Outcome", axis=1).values
    x_elements = x.shape[0]
    x = list(x)
    y = list(data["Outcome"].values.reshape(-1, 1))

    i = 0
    deleted_elements_count = 0

    while i < x_elements - deleted_elements_count:
        for j in range(1, 6):
            if x[i][j] == 0:
                x.pop(i)
                y.pop(i)
                i -= 1
                deleted_elements_count += 1
                break
        i += 1

    x = np.array(x, "float32")
    y = np.array(y, "float32")

    x = StandardScaler(x)

    a = np.sum(np.float32(y == 0))

    print(f"0: {(a / len(y)) * 100}%")
    print(f"1: {100 - (a / len(y)) * 100}%")

    np.savez(r"Datasets\diabetes.npz", x=x, y=y)'''

    data = np.load(r"Datasets\diabetes.npz")
    x = data["x"]
    y = data["y"]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    return x_train, x_test, y_train, y_test