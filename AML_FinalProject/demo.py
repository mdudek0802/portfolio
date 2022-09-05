import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from warnings import simplefilter

CSV = 'water_potability.csv'

DATA = pd.read_csv(CSV)
DATA.dropna(inplace=True)

FEATURES = DATA.iloc[:, :len(DATA.columns) - 1]
FEATURES = (FEATURES - FEATURES.min())/(FEATURES.max() - FEATURES.min())

DATA.iloc[:, :len(DATA.columns) - 1] = FEATURES

# ignore all warnings
simplefilter(action='ignore')

pd.set_option('display.max_columns', 50)

BEST_LOG_REG_PARAMS = [0.9, 89, 'l2', 'saga'] # training size, c-value, regression, and optimization algorithm
BEST_SVM_PARAMS = [0.9, 89, 89]               # training size, c-value, and gamma
BEST_NEURAL_NET_PARAMS = [0.3, [13, 13, 13], 'logistic', 'sgd', 34] # training size, hidden-layers, activation function, solver algorithm, and c-value

stat_plot, stat_axes = plt.subplots(len(DATA.columns))

def plot_by_one_feature(plot_num, feature_name, feature1_potable, feature2_non_potable):
    stat_axes[plot_num].scatter(list(range(len(feature1_potable))), feature1_potable)
    stat_axes[plot_num].scatter(list(range(len(feature2_non_potable))), feature2_non_potable)

if __name__ == "__main__":

    for i in range(len(DATA.columns)):
        plot_by_one_feature(i, DATA.columns[i], (DATA.loc[DATA['Potability'] == 0]).iloc[:, i], (DATA.loc[DATA['Potability'] == 1]).iloc[:, i])

    stat_plot.show()
    log_reg_train, log_reg_test = train_test_split(DATA, train_size=BEST_LOG_REG_PARAMS[0])
    best_log_reg_model = LogisticRegression(C=BEST_LOG_REG_PARAMS[1], penalty=BEST_LOG_REG_PARAMS[2], solver=BEST_LOG_REG_PARAMS[3], max_iter=1000)
    best_log_reg_model.fit(log_reg_train.iloc[:, 0:(len(DATA.columns)-1)], log_reg_train.iloc[:, (len(DATA.columns)-1):])

    svm_train, svm_test = train_test_split(DATA, train_size=BEST_SVM_PARAMS[0])
    best_svm_model = SVC()
    best_svm_model.fit(svm_train.iloc[:, 0:(len(DATA.columns)-1)], svm_train.iloc[:, (len(DATA.columns)-1):])

    neural_net_train, neural_net_test = train_test_split(DATA, train_size=BEST_NEURAL_NET_PARAMS[0])
    best_neural_net_model = MLPClassifier()
    best_neural_net_model.fit(neural_net_train.iloc[:, 0:(len(DATA.columns)-1)], neural_net_train.iloc[:, (len(DATA.columns)-1):])

    log_reg_conf_matrix = ConfusionMatrixDisplay.from_estimator(
            best_log_reg_model,
            log_reg_test.iloc[:, 0:(len(DATA.columns)-1)],
            log_reg_test.iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    log_reg_conf_matrix.ax_.set_title("Log Reg Confusion Matrix")

    svm_conf_matrix = ConfusionMatrixDisplay.from_estimator(
            best_svm_model,
            svm_test.iloc[:, 0:(len(DATA.columns)-1)],
            svm_test.iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    svm_conf_matrix.ax_.set_title("SVM Confusion Matrix")

    neural_net_conf_matrix = ConfusionMatrixDisplay.from_estimator(
            best_neural_net_model,
            neural_net_test.iloc[:, 0:(len(DATA.columns)-1)],
            neural_net_test.iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    neural_net_conf_matrix.ax_.set_title("Neural Nets Confusion Matrix")

    plt.show(block=True)
