import numpy as np
import pandas as pd
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



print("Median of attributes:\n", median(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMin of attributes:\n", min(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nMax of attributes:\n", max(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]))
print("\nDominant: ", dominant(data["class"]))
