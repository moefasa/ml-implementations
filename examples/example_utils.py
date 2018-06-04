import os
import sys

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (14,8)

def get_errors(model, X_train, y_train, X_test, y_test, beta=None):

    train_preds = model.predict(X_train, beta=beta)
    test_preds = model.predict(X_test, beta=beta)

    train_error = 1 - accuracy_score(y_train, train_preds)
    test_error = 1 - accuracy_score(y_test, test_preds)

    return train_error, test_error

def plot_errors(model, X_train, y_train, X_test, y_test):
    beta_vals = model.betas_
    num_iters = len(beta_vals)
    train_errors = []
    test_errors = []
    for i in range(num_iters):
        train_error, test_error = get_errors(model, X_train, y_train, X_test, y_test, beta=beta_vals[i])
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.plot(range(num_iters), train_errors, label="Training Error", color="blue")
    plt.plot(range(num_iters), test_errors, label="Validation Error", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.show()
