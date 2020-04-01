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

def test_normal_distribution(p_group, alpha)
    return p_group < alpha

def test_hipothesis(p, alpha):
    return p > alpha

W, p_female = st.shapiro(df_female["Hwt"])
print('For female cats normal distribution test result is:', test_normal_distribution(p_female, alpha))
W, p_male = st.shapiro(df_male["Hwt"])
print('For female cats normal distribution test result is:', test_normal_distribution(p_male, alpha))

t, p = st.ttest_ind(df_female["Hwt"], df_male["Hwt"])
print ("Hipothesis that Heart weight for male and female heart is equal is: ", test_hipothesis(p, alpha))

def display_hist(data_female, data_male):
    data.plot.hist(bins=40)
    plt.legend(loc="upper right")
    plt.xlabel('value [kg]')
    plt.title("Cats Histogram")
    plt.show()

def create_histogram(df_m, df_f, min_value, max_value, to_compare, result, standard_devaition, hypothesis_result):
    min_interval = min_value
    max_interval = max_value
    n_f, bins_f, patches = py.hist(df_f["Hwt"], 100, density=True, facecolor='green', alpha=0.75)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = st.norm.pdf( bincenters, loc = result, scale = standard_deviation)
    l = py.plot(bincenters, y, 'r--', linewidth=1)
    plt.axvline(to_compare, color='red', label="hypothesis value")
    plt.axvline(min_interval, color='red', label="hypothesis value margin", linestyle='dashed')
    plt.axvline(max_interval, color='red', linestyle='dashed')
    plt.axvspan(min_interval, max_interval, alpha=0.2, color='red')
    plt.axvline(min_interval, color='orange', label="calculated value margin", linestyle='dashed')
    plt.axvline(max_interval, color='orange', linestyle='dashed')
    plt.axvspan(min_interval, max_interval, alpha=0.2, color='orange')
    plt.axvline(result, color='orange', label="calculated value")
    plt.legend(loc="upper left", fontsize="x-small")
    plt.text(value_to_compare, 0.5, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5), fontsize="x-small")
    plt.text(value_to_compare, 0, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5), fontsize="x-small")
    plt.xlabel('value [kg]')
    plt.title("Cats Histogram")
     plt.show()