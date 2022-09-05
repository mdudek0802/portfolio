import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, precision_score
from sklearn import preprocessing
from warnings import simplefilter
import multiprocessing as mp

CSV = 'water_potability.csv'

DATA = pd.read_csv(CSV)
DATA.dropna(inplace=True)
# DATA.drop(['Solids','Conductivity', 'Sulfate'], axis=1, inplace=True)

FEATURES = DATA.iloc[:, :len(DATA.columns) - 1]
FEATURES = (FEATURES - FEATURES.min())/(FEATURES.max() - FEATURES.min())

DATA.iloc[:, :len(DATA.columns) - 1] = FEATURES

# ignore all warnings
simplefilter(action='ignore')

REGULARIZATIONS = {
    0 : 'none',
    1 : 'l1',
    2 : 'l2',
}

OPTIMIZATION_ALGORITHMS_NO_REG = {
    0 : 'newton-cg',
    1 : 'lbfgs',
    2 : 'sag',
    3 : 'saga',
}

OPTIMIZATION_ALGORITHMS_L1_REG = {
    0 : 'liblinear',
    1 : 'saga',
}

OPTIMIZATION_ALGORITHMS_L2_REG = {
    0 : 'newton-cg',
    1 : 'lbfgs',
    2 : 'liblinear',
    3 : 'sag',
    4 : 'saga',
}

OPTIMIZATIONS_BY_REG = {
    0 : OPTIMIZATION_ALGORITHMS_NO_REG,
    1 : OPTIMIZATION_ALGORITHMS_L1_REG,
    2 : OPTIMIZATION_ALGORITHMS_L2_REG,
}

TRAINING_SIZES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

C_VALS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

pd.set_option('display.max_columns', 50)

def runAccuracies(training_size):
    accuracies = []
    train, test = train_test_split(DATA, train_size=training_size)

    for c in range(len(C_VALS)):
            for reg in range(len(REGULARIZATIONS)):
                for alg in range(len(OPTIMIZATIONS_BY_REG[reg])):
                    model = LogisticRegression(C=C_VALS[c], solver=OPTIMIZATIONS_BY_REG[reg][alg], max_iter=1000)
                    model.fit(train.iloc[:, 0:(len(DATA.columns)-1)], train.iloc[:, (len(DATA.columns)-1):])
                    acc = model.score(test.iloc[:, 0:(len(DATA.columns)-1)], test.iloc[:, (len(DATA.columns)-1):])
                    accuracies.append([training_size, C_VALS[c], REGULARIZATIONS[reg], OPTIMIZATIONS_BY_REG[reg][alg], acc, train, test, model])

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
        if (best_prec < precision_score(accs[i][6].iloc[:, (len(DATA.columns)-1):], (accs[i][7]).predict(accs[i][6].iloc[:, 0:(len(DATA.columns)-1)]))):
            best_acc_index = i

    print(accs[best_acc_index][:5])

    disp = ConfusionMatrixDisplay.from_estimator(
            accs[best_acc_index][7],
            accs[best_acc_index][6].iloc[:, 0:(len(DATA.columns)-1)],
            accs[best_acc_index][6].iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    disp.ax_.set_title("Confusion Matrix")

    plt.show(block=True)
