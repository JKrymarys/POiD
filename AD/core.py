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
    # col_correlations = data.corr()
    # col_correlations.loc[:, :] = np.tril(col_correlations, k=-1)
    # cor_pairs = col_correlations.stack()
    au_corr = df.corr().abs().unstack()

    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    
    au_corr = au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    return au_corr[0:1]

def hist(df, cols):
    df.hist(column=cols)


print("Median of attributes:\n", median(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMin of attributes:\n", min(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMax of attributes:\n", max(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nDominant: ", dominant(data["class"]))

#Narysować histogramy dla dwóch cech ilościowych najbardziej ze sobą skorelowanych

print("\nCorrelations:\n", data.corr())

print("\nMost correlated pair:\n", correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))

cor_column_names = correlation_data(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]).index[0]

#hist(data, cor_column_names[0])
	
data['petal_length'].plot.hist(bins=100)
plt.show()