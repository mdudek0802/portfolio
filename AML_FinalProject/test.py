import pandas as pd
import numpy as np

CSV = 'water_potability.csv'

pd.set_option('display.max_columns', 50)

if __name__ == "__main__":
    frame = pd.read_csv(CSV)
    frame.dropna(inplace=True)

    potables = frame[frame['Potability'] == 1]
    nonpotables = frame[frame['Potability'] == 0]

    print("Potables:\nMax")
    print(potables.max())
    print('\nMin')
    print(potables.min())
    print('\nMean')
    print(potables.mean())
    print('\nMedian')
    print(potables.median())
    print('\nMode')
    print(potables.round().mode())
    print('\nStandard Deviation')
    print(potables.std())

    print('\n\n')

    print("Nonpotables:\nMax")
    print(nonpotables.max())
    print('\nMin')
    print(nonpotables.min())
    print('\nMean')
    print(nonpotables.mean())
    print('\nMedian')
    print(nonpotables.median())
    print('\nMode')
    print(nonpotables.round().mode())
    print('\nStandard Deviation')
    print(nonpotables.std())
