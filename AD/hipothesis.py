# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import scipy.stats as st
import pylab as py
import matplotlib.pyplot as plt

#1A. Dla danych Births zbadać hipotezę, że dzienna średnia liczba urodzeń dzieci wynosi: 10000.
#1B. Dla danych manaus zbadać hipotezę, że średnia wysokość rzeki w manaus jest na wysokości punktu arbitralnego(wynosi 0).
#1C. Dla danych quakes zbadać hipotezę, że średnia głębokość występowania trzęsienia ziemi wynosi 300 metrów. 
#2. Zwizualizowac rozklady na histogramie.
#3. Zaznaczyć na wykresie punkt dotyczacy badanej hipotezy.

#define statistical significance
alpha = 0.05 

# df = pd.read_csv('datasets/Births.csv', sep=",", index_col=0)
# value_to_compare = 10000
# key = "births"

df = pd.read_csv('datasets/manaus.csv', sep=",", index_col=0)
key = "manaus"
value_to_compare = 0

# df = pd.read_csv('datasets/quakes.csv', sep=",", index_col=0)
# key = "depth"
# value_to_compare = 300

#print(df)

def get_sample_no(df):
    return df.shape[0]

def calculate_average(df, key, samples):
    values_sum =0
    for row in range(samples):
        values_sum += df[key].iloc[row]
    return values_sum/df.shape[0]

def calculate_standard_deviation(df, key):
    return df[key].std()
       
def calculate_coef(standard_deviation, samples):
    return 1.9600*(standard_deviation/np.sqrt(samples))

def test_hypothesis(min_value, max_value, to_compare):
    return min_value <= to_compare <= max_value

def create_histogram(col, min_value, max_value, to_compare, result, standard_devaition, hypothesis_result):
    min_interval = min_value
    max_interval = max_value
    n, bins, patches = py.hist(col, 100, density=True, facecolor='green', alpha=0.75)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = st.norm.pdf( bincenters, loc = result, scale = standard_deviation)
    l = py.plot(bincenters, y, 'r--', linewidth=1)
    plt.axvline(to_compare, color='red', label="hypothesis value")
    plt.axvline(min_interval, color='orange', label="calculated value margin", linestyle='dashed')
    plt.axvline(max_interval, color='orange', linestyle='dashed')
    plt.axvspan(min_interval, max_interval, alpha=0.2, color='orange')
    plt.axvline(result, color='orange', label="calculated value")
    plt.legend(loc="upper left", fontsize="x-small")
    plt.text(value_to_compare, 0, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5), fontsize="x-small")
    plt.xlabel('depth')
    plt.title("Histogram")
    plt.show()

def histfit(col):  
    n, bins, patches = py.hist(col, 100, density=True, facecolor='green', alpha=0.75)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = st.norm.pdf( bincenters, loc = np.mean(col), scale = np.std(col))
    l = py.plot(bincenters, y, 'r--', linewidth=1)
    plt.show()

sample_no = get_sample_no(df)
mean_value = calculate_average(df, key, sample_no)
standard_deviation = calculate_standard_deviation(df, key)
coef = calculate_coef(standard_deviation, sample_no)
min_value = mean_value - coef
max_value = mean_value + coef

hypothesis_result = test_hypothesis(min_value, max_value, value_to_compare)

print("Average {key}: ", mean_value)
print("standard deviation: ", standard_deviation)
create_histogram(df[key], min_value, max_value, value_to_compare, mean_value, standard_deviation, hypothesis_result)
#histfit(df[key])



