import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Dla poszczególnych atrybutów wyznaczyć medianę, minimum i maximum dla cech ilościowych i dominantę dla cech jakościowych

data = pd.read_csv('iris.data', sep=",", header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

def calculate_median(df):
    return df.median(axis=0)

def get_min(df):
    return df.min(axis=0)

def get_max(df):
    return df.max(axis=0)

def get_dominant(col):
    return col.mode().iloc[0]

def correlation_data(df):
    au_corr = df.corr().abs().unstack()

    #get pairs to drop - those are pair on the diagonal that have correlation equal to 1
    pairs_to_drop = set()
    cols = df.columns
    number_of_cols = df.shape[1]
    for i in range(0, number_of_cols):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    au_corr = au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return au_corr[0:1]

def display_hist(data, cols):
    data[cols[0]].plot.hist(bins=40)
    data[cols[1]].plot.hist(bins=40)
    plt.legend(loc="upper right")
    plt.xlabel('value [cm]')
    plt.title("Iris Histogram")
    plt.show()

print("Median of attributes:\n\n", calculate_median(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMin of attributes:\n\n", get_min(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMax of attributes:\n\n", get_max(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nDominant: ", get_dominant(data["class"]))

#Narysować histogramy dla dwóch cech ilościowych najbardziej ze sobą skorelowanych

#Print correlations for entire dataset
print("\nCorrelations:\n\n", data.corr())

print("\nMost correlated pair:\n\n", correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))

cor_column_names = correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]).index[0]
display_hist(data, cor_column_names)	