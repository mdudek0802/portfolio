import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, precision_score
from warnings import simplefilter

CSV = 'water_potability.csv'
DATA = pd.read_csv(CSV)
DATA.dropna(inplace=True)

FEATURES = DATA.iloc[:, :len(DATA.columns) - 1]
FEATURES = (FEATURES - FEATURES.min())/(FEATURES.max() - FEATURES.min())
DATA.iloc[:, :len(DATA.columns) - 1] = FEATURES

# ignore all warnings
simplefilter(action='ignore')

TRAINING_SIZES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
C_VALUES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
GAMMA_VALUES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

pd.set_option('display.max_columns', 50)

def runAccuracies(training_size):
    accuracies = []
    train, test = train_test_split(DATA, train_size=training_size)

    for c_val in C_VALUES:
        for g_val in GAMMA_VALUES:
            model = SVC(C=c_val, gamma=g_val)
            model.fit(train.iloc[:, 0:(len(DATA.columns)-1)], np.ravel(train.iloc[:, (len(DATA.columns)-1):]))
            acc = model.score(test.iloc[:, 0:(len(DATA.columns)-1)], test.iloc[:, (len(DATA.columns)-1):])
            accuracies.append([training_size, c_val, g_val, acc, train, test, model])

    return accuracies

def getAccuracies():
    accuracies = []

    with mp.Pool(len(TRAINING_SIZES)) as p:
        accuracies_list = p.map(runAccuracies, TRAINING_SIZES)

    for acc in accuracies_list:
        accuracies.extend(acc)

    return accuracies

if __name__ == "__main__":
    accs = getAccuracies()

    best_acc_index = 0
    best_prec = 0
    for i in range(len(accs)):
        if (best_prec < precision_score(accs[i][5].iloc[:, (len(DATA.columns)-1):], (accs[i][6]).predict(accs[i][5].iloc[:, 0:(len(DATA.columns)-1)]))):
            best_acc_index = i

    print(accs[best_acc_index][:4])

    disp = ConfusionMatrixDisplay.from_estimator(
            accs[best_acc_index][6],
            accs[best_acc_index][5].iloc[:, 0:(len(DATA.columns)-1)],
            accs[best_acc_index][5].iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    disp.ax_.set_title("Confusion Matrix")

    plt.show(block=True)
