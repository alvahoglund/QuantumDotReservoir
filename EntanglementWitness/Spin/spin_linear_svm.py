import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    train_measurements = np.load(train_data_path)[:, :]
    train_labels = np.load(train_labels_path)
    test_measurements = np.load(test_data_path)[:, :]
    test_labels = np.load(test_labels_path)
    return train_measurements, train_labels, test_measurements, test_labels

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_svm(X_train_scaled, y_train):
    clf = LinearSVC(penalty='l2', class_weight={-1: 1, 1: 1000}, C = 1000)
    clf.fit(X_train_scaled, y_train)
    return clf

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    accuracy_sep = np.mean(y_pred[y == 1] == 1)
    accuracy_ent = np.mean(y_pred[y == -1] == -1)
    return accuracy, accuracy_sep, accuracy_ent, y_pred

def rescale_weights(w_scaled, b_scaled, scaler):
    w = w_scaled / scaler.scale_
    b = b_scaled - np.dot(w_scaled, scaler.mean_ / scaler.scale_)
    return w, b

def print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test):
    accuracy, accuracy_sep, accuracy_ent, y_pred = evaluate_model(clf, X_train_scaled, y_train)
    print(f"Training Accuracy: {accuracy}")
    print(f"Training Accuracy (Separable States): {accuracy_sep}")
    print(f"Training Accuracy (Entangled States): {accuracy_ent}")

    accuracy, accuracy_sep, accuracy_ent, y_pred = evaluate_model(clf, X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy (Separable States): {accuracy_sep}")
    print(f"Test Accuracy (Entangled States): {accuracy_ent}")

def plot_data(X, y, w, b, title = ""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50))

    #Plot the decision boundary
    if w[2] != 0:
        x3 = -(w[0] * x1 + w[1] * x2 + b) / w[2]
    else:
        raise ValueError("The weight corresponding to the third feature is zero, so the decision boundary cannot be plotted as a plane.")

    ax.plot_surface(x1, x2, x3, color='lightblue', alpha=0.5, edgecolor='k', label='Decision Boundary')

    # Plot the measurements
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm_r, s=50, edgecolors='k', marker = 'o')


    class_1_label = 'Separable States'
    class_2_label = 'Entangled States'
    handle_class_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm_r(0.99), markersize=10, label=class_1_label)
    handle_class_2 = Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm_r(0), markersize=10, label=class_2_label)
    ax.legend(handles=[handle_class_1, handle_class_2], loc='best')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train, X_test, y_test = load_data("EntanglementWitness\Spin\Data\measurements_train.npy", "EntanglementWitness\Spin\Data\labels_train.npy", "EntanglementWitness\Spin\Data\measurements_test.npy", "EntanglementWitness/Spin\Data\labels_test.npy")
    
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)

    clf = train_svm(X_train_scaled, y_train)

    print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test)

    w_original, b_original = rescale_weights(clf.coef_[0], clf.intercept_[0], scaler)

    plot_data(X_train, y_train,w_original, b_original, r"EW for $|\uparrow \downarrow> - | \downarrow \uparrow > $")

if __name__ == "__main__":
    main()