from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

DATASET = "2015.csv"
K_NEIGHBORS = 7

def plot_by_three_attributes(regions, attrib_1, attrib_2, attrib_3):
    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')
    i = 0
    for region in regions:
        ax.scatter(region[attrib_1], region[attrib_2], region[attrib_3])
        i+=1

    plt.xlabel(attrib_1)
    plt.ylabel(attrib_2)
    ax.set_zlabel(attrib_3)
    ax.legend(unique_regions, ncol=2, fontsize='small')
    plt.title(attrib_1 + " vs " + attrib_2 + " vs " + attrib_3)

    plt.show()

def plot_by_two_attributes(regions, attrib_1, attrib_2):
    fig = plt.figure()
    ax = fig.add_subplot()
    i = 0
    for region in regions:
        ax.scatter(region[attrib_1], region[attrib_2])
        i+=1

    ax.legend(unique_regions, ncol=2, fontsize='small')
    plt.xlabel(attrib_1)
    plt.ylabel(attrib_2)
    plt.title(attrib_1 + " vs " + attrib_2)

    plt.show()

if __name__ == "__main__":
    unique_regions = []
    region_density = []
    region_data = []
    data = pd.read_csv(DATASET)
    data = pd.DataFrame(data)
    regions = data["Region"]
    for region in regions:
        if region not in unique_regions and (region != 'North America' and region != 'Australia and New Zealand'):
            unique_regions.append(region)
    
    # print(len(unique_regions))
    # print(unique_regions)

    sub_data = data[["Region", "Happiness Score", "Economy (GDP per Capita)",
                    "Family", "Health (Life Expectancy)", "Freedom",
                    "Trust (Government Corruption)", "Generosity"]]

    print("# of attributes: " + str(len(sub_data.columns) - 1))

    sub_data = sub_data[sub_data["Region"] != 'North America']
    sub_data = sub_data[sub_data["Region"] != 'Australia and New Zealand']

    for region in unique_regions:
        temp_region_data = sub_data[sub_data["Region"] == region]
        if len(temp_region_data) <= 5:
            continue
        region_density.append(len(temp_region_data))
        region_data.append(temp_region_data)

    # Uncomment to show a plot of different attributes
    plot_by_three_attributes(region_data, "Happiness Score", "Economy (GDP per Capita)", "Health (Life Expectancy)")
    plot_by_three_attributes(region_data, "Family", "Freedom", "Trust (Government Corruption)")
    # plot_by_two_attributes(region_data, "Happiness Score", "Economy (GDP per Capita)")

    i = 0
    for region in unique_regions:
        print("Class: " + region + " has " + str(region_density[i]) + " instances")
        i += 1
    print("For a total of: " + str(sum(region_density)) + " instances")

    train_size=0.7
    print("Using: " + str(train_size * 100) + "% for training.")

    # Use commented out version for reproducable splitage
    # regions_train, regions_test = train_test_split(sub_data, train_size=train_size, random_state=0)
    regions_train, regions_test = train_test_split(sub_data, train_size=train_size)

    metric = "euclidean"
    print("Using: " + str(metric) + " for the distance metric")

    classifier = KNeighborsClassifier(K_NEIGHBORS, metric=metric)
    classifier.fit(regions_train.iloc[:, 1:], regions_train.iloc[:, 0])

    predicted = classifier.predict(regions_test.iloc[:, 1:])

    disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            regions_test.iloc[:, 1:],
            regions_test.iloc[:, 0],
            # display_labels=unique_regions,
            cmap=plt.cm.Blues,
            normalize=None,
        )
    disp.ax_.set_title("Confusion Matrix")

    plt.show()

