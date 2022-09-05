import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, precision_score
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

HIDDEN_LAYERS = {
    0 :  [8, 8],
    1 :  [13, 13],
    2 :  [21, 21],
    3 :  [34, 34],
    4 :  [55, 55],
    5 :  [89, 89],
    6 :  [100, 100],
    7 :  [8, 8, 8],
    8 :  [13, 13, 13],
    9 :  [21, 21, 21],
    10 : [34, 34, 34],
    11 : [55, 55, 55],
    12 : [89, 89, 89],
    13 : [100, 100, 100],
}

ACTIVATION_FUNCTIONS = {
    0 : 'identity',
    1 : 'logistic',
    2 : 'tanh',
    3 : 'relu',
}

OPTIMIZATION_ALGORITHMS = {
    0 : 'sgd',
    1 : 'adam',
}

TRAINING_SIZES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

ALPHA_VALS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

pd.set_option('display.max_columns', 50)

def runAccuracies(training_size):
    accuracies = []
    train, test = train_test_split(DATA, train_size=training_size)

    for i in range(len(HIDDEN_LAYERS)):
        for c in range(len(ALPHA_VALS)):
                for func in range(len(ACTIVATION_FUNCTIONS)):
                    for alg in range(len(OPTIMIZATION_ALGORITHMS)):
                        model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS[i], activation=ACTIVATION_FUNCTIONS[func], solver=OPTIMIZATION_ALGORITHMS[alg], alpha=ALPHA_VALS[c])
                        model.fit(train.iloc[:, 0:(len(DATA.columns)-1)], train.iloc[:, (len(DATA.columns)-1):])
                        acc = model.score(test.iloc[:, 0:(len(DATA.columns)-1)], test.iloc[:, (len(DATA.columns)-1):])
                        accuracies.append([training_size, HIDDEN_LAYERS[i], ACTIVATION_FUNCTIONS[func], OPTIMIZATION_ALGORITHMS[alg], ALPHA_VALS[c], acc, train, test, model])

    print("Process with training size: " + str(training_size) + " is done!\n\n")
    return accuracies

def getAccuracies():
    accuracies = []

    with mp.Pool(len(TRAINING_SIZES)) as p:
        accuracies_list = p.map(runAccuracies, TRAINING_SIZES)

    for acc in accuracies_list:
        accuracies.extend(acc)

    return accuracies


if __name__ == "__main__":
    start_time = time.time()
    accs = getAccuracies()
    
    best_acc_index = 0
    best_prec = 0
    for i in range(len(accs)):
        if (best_prec < precision_score(accs[i][7].iloc[:, (len(DATA.columns)-1):], (accs[i][8]).predict(accs[i][7].iloc[:, 0:(len(DATA.columns)-1)]))):
            best_acc_index = i

    print(accs[best_acc_index][:6])
    end_time = time.time()

    disp = ConfusionMatrixDisplay.from_estimator(
            accs[best_acc_index][8],
            accs[best_acc_index][7].iloc[:, 0:(len(DATA.columns)-1)],
            accs[best_acc_index][7].iloc[:, (len(DATA.columns)-1):],
            cmap=plt.cm.Blues,
            normalize=None,
        )

    disp.ax_.set_title("Confusion Matrix")

    plt.show(block=True)

    print("Took: " + str(end_time - start_time) + " seconds!")
