import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#1A. Dla danych Births zbadać hipotezę, że dzienna średnia liczba urodzeń dzieci wynosi: 10000.
#2. Zwizualizować rozkłady na histogramie.
#3. Zaznaczyć na wykresie punkt dotyczący badanej hipotezy.

df = pd.read_csv('Births.csv', sep=",", index_col=0)
value_to_compare = 10000

print(df)
births_sum =0
#calculate average daily births
for row in range(df.shape[0]):
    births_sum += df["births"].iloc[row]

def test_hypothesis(to_compare, result):
    min_interval = to_compare - (to_compare*0.05)
    max_interval = to_compare + (to_compare*0.05)
    if min_interval <= result <= max_interval:
        return True
    else:
        return False

def hist(col, to_compare, result, hipothesis_result):
    min_interval = to_compare - (to_compare*0.05)
    max_interval = to_compare + (to_compare*0.05)
    col.plot.hist(bins=100, color='green')
    plt.axvline(to_compare, color='red', label="hypothesis value")
    plt.axvline(min_interval, color='red', label="hypothesis value margin", linestyle='dashed')
    plt.axvline(max_interval, color='red', linestyle='dashed')
    plt.axvline(result, color='orange', label="calculated value")
    plt.legend(loc="upper left")
    plt.text(6500, 110, f"Hypothesis result: {hypothesis_result}", bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('number of births')
    plt.title("Daily Births Histogram")
    plt.show()

result_calculated = births_sum/df.shape[0]
result_funct = df["births"].mean(axis=0)
hypothesis_result = test_hypothesis(value_to_compare, result_funct)
hist(df["births"], value_to_compare, result_funct, hypothesis_result)

print("Average daily births (calculated): ", result_calculated)
print("Average daily births (from function): ", result_funct)
print("Hypothesis result: ", hypothesis_result)



