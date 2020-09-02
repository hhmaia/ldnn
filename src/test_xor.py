import numpy as np
from src.ldnn import l_layer_model_train, l_model_forward
from src.ldnn_utils import plot_costs

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float').T
y_train = np.array([0, 1, 1, 0], dtype='float').reshape((1, 4))

params, costs = l_layer_model_train(x_train, y_train, [2, 10, 1], epochs=10000, print_costs=True)
yhat, _ = l_model_forward(x_train, params)
plot_costs(costs)

print('exiting')