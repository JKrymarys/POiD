import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import pylab as py

#define statistical significance
alpha = 0.05

df = pd.read_csv('datasets/cats-data.csv', sep=",", index_col=0)
print(df)

df_female = df[df["Sex"] == "F"]
df_male = df[df["Sex"] == "M"]

def test_normal_distribution(p_group, alpha):
    return p_group > alpha

def test_hipothesis(p, alpha):
    return p > alpha

W, p_female = st.shapiro(df_female["Hwt"])
print('For female cats normal distribution test result is:', test_normal_distribution(p_female, alpha))
W, p_male = st.shapiro(df_male["Hwt"])
print('For male cats normal distribution test result is:', test_normal_distribution(p_male, alpha))

t, p = st.ttest_ind(df_female["Hwt"], df_male["Hwt"])
hypothesis_result = test_hipothesis(p, alpha)
print ("Hipothesis that Heart weight for male and female heart is equal is: ", hypothesis_result)

def display_hist(data_female, data_male):
    data.plot.hist(bins=40)
    plt.legend(loc="upper right")
    plt.xlabel('value [kg]')
    plt.title("Cats Histogram")
    plt.show()

def create_histogram(df_m, df_f, hypothesis_result):
    n_f, bins_f, patches_f = py.hist(df_f["Hwt"], 10, density=True, facecolor='green', alpha=0.75, label='Female Cats')
    bincenters_f = 0.5*(bins_f[1:]+bins_f[:-1])
    y_f = st.norm.pdf( bincenters_f, loc = df_f["Hwt"].mean(axis=0), scale = df_f["Hwt"].std())
    l_f = py.plot(bincenters_f, y_f, 'r--', linewidth=1, color='green')
    n_m, bins_m, patches_m = py.hist(df_m["Hwt"], 10, density=True, facecolor='blue', alpha=0.75, label='Male Cats')
    bincenters_m = 0.5*(bins_m[1:]+bins_m[:-1])
    y_m = st.norm.pdf( bincenters_m, loc = df_m["Hwt"].mean(axis=0), scale = df_m["Hwt"].std())
    l_m = py.plot(bincenters_m, y_m, 'r--', linewidth=1, color='blue')
    plt.legend(['Female cats', "Male cats"], loc="upper left", fontsize="x-small")
    plt.text(16, 0.2, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5), fontsize="x-small")
    plt.xlabel('Hwt [kg]')
    plt.title("Cats Histogram")
    plt.show()

create_histogram(df_male, df_female, hypothesis_result)