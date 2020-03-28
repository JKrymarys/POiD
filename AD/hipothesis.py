# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1A. Dla danych Births zbadać hipotezę, że dzienna średnia liczba urodzeń dzieci wynosi: 10000.
#1B. Dla danych manaus zbadać hipotezę, że średnia wysokość rzeki w manaus jest na wysokości punktu arbitralnego(wynosi 0).
#1C. Dla danych quakes zbadać hipotezę, że średnia głębokość występowania trzęsienia ziemi wynosi 300 metrów. 
#2. Zwizualizowac rozklady na histogramie.
#3. Zaznaczyć na wykresie punkt dotyczacy badanej hipotezy.

# df = pd.read_csv('datasets/Births.csv', sep=",", index_col=0)
# value_to_compare = 10000
# key = "births"

df = pd.read_csv('datasets/manaus.csv', sep=",", index_col=0)
key = "manaus"
value_to_compare = 0

# df = pd.read_csv('datasets/quakes.csv', sep=",", index_col=0)
# key = "depth"
# value_to_compare = 300



print(df)
values_sum =0

for row in range(df.shape[0]):
    values_sum += df[key].iloc[row]


def test_hypothesis(to_compare, result):
    min_interval = to_compare - (to_compare*0.05)
    max_interval = to_compare + (to_compare*0.05)
    return min_interval <= result <= max_interval
  

def create_histogram(col, to_compare, result, hipothesis_result):
    min_interval = to_compare - (to_compare*0.05)
    max_interval = to_compare + (to_compare*0.05)
    col.plot.hist(bins=100, color='green')
    plt.axvline(to_compare, color='red', label="hypothesis value")
    plt.axvline(min_interval, color='red', label="hypothesis value margin", linestyle='dashed')
    plt.axvline(max_interval, color='red', linestyle='dashed')
    plt.axvspan(min_interval, max_interval, alpha=0.2, color='red')
    plt.axvline(result, color='orange', label="calculated value")
    plt.legend(loc="upper left", fontsize="x-small")
    plt.text(value_to_compare, 0.5, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5), fontsize="x-small")
    plt.xlabel('depth')
    plt.title("Histogram")
    plt.show()


result_calculated = values_sum/df.shape[0]
print("result_calculated", result_calculated)

result_funct = df[key].mean(axis=0)
hypothesis_result = test_hypothesis(value_to_compare, result_funct)
create_histogram(df[key], value_to_compare, result_funct, hypothesis_result)

print(f"Average {key}  (calculated): ", result_calculated)
print(f"Average {key} (from function): ", result_funct)
print("Hypothesis result: ", hypothesis_result)



