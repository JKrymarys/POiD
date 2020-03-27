import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Dla poszczególnych atrybutów wyznaczyć medianę, minimum i maximum dla cech ilościowych i dominantę dla cech jakościowych

data = pd.read_csv('iris.data', sep=",", header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

def median(df):
    median = df.median(axis=0)
    return median

def min(df):
    min = df.min(axis=0)
    return min

def max(df):
    max = df.max(axis=0)
    return max

def dominant(col):
    dominant = col.mode().iloc[0]
    return dominant

def correlation_data(df):
    au_corr = df.corr().abs().unstack()

    #get pairs to drop - those are pair on the diagonal that have correlation equal to 1
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    au_corr = au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return au_corr[0:1]

def hist(data, cols):
    data[cols[0]].plot.hist(bins=40)
    data[cols[1]].plot.hist(bins=40)
    plt.legend(loc="upper right")
    plt.xlabel('value [cm]')
    plt.title("Iris Histogram")
    plt.show()

print("Median of attributes:\n\n", median(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMin of attributes:\n\n", min(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMax of attributes:\n\n", max(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nDominant: ", dominant(data["class"]))

#Narysować histogramy dla dwóch cech ilościowych najbardziej ze sobą skorelowanych

#Print correlations for entire dataset
print("\nCorrelations:\n\n", data.corr())

print("\nMost correlated pair:\n\n", correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))

cor_column_names = correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]).index[0]
hist(data, cor_column_names)	