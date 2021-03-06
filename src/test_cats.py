import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.ldnn import l_model_forward, l_layer_model_train
from src.ldnn_utils import plot_costs

def load_data():
    train_dataset = h5py.File(os.getcwd().replace('src', '') + '/data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(os.getcwd().replace('src', '') + '/data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def predict(X, Y, parameters):
    assert X.shape[1] == Y.shape[1]
    assert parameters[1]['W'].shape[1] == X.shape[0]
    m = X.shape[1]
    # Forward propagation
    AL, _ = l_model_forward(X, parameters)
    # mapping probabilities outputted by the model to predictions
    predictions = np.array(list(map(lambda x: 1 if x>0.5 else 0, AL.squeeze())))
    print("Accuracy: " + str(np.sum((predictions == Y) / m)))
    return predictions


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x = np.reshape(train_x_orig, (train_x_orig.shape[0], -1)).T/255.
test_x = np.reshape(test_x_orig, (test_x_orig.shape[0], -1)).T/255.

params, costs = l_layer_model_train(train_x, train_y,
                    [train_x.shape[0], 20, 20, 20, 1],
                    epochs=1500,
                    learning_rate=0.002,
                    batch_size=32,
                    l2_lambda=0.01,
                    print_costs=True)
plot_costs(costs)

predictions = predict(test_x, test_y, params)

print('exiting...')
