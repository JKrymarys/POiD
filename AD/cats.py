import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import core


df = pd.read_csv('datasets/cats-data.csv', sep=",", index_col=0)
print(df)

df_female = df[df["Sex"] == "F"]
df_male = df[df["Sex"] == "M"]

def display_hist(data_female, data_mal):
    data.plot.hist(bins=40)
    plt.legend(loc="upper right")
    plt.xlabel('value [kg]')
    plt.title("Cats Histogram")
    plt.show()



print("Median, min and max weight for female cats: ", core.calculate_median(df_female["Bwt"]), core.get_min(df_female["Bwt"]), core.get_max(df_female["Bwt"]))
print("Median, min and max weight for malecats: ", core.calculate_median(df_male["Bwt"]), core.get_min(df_male["Bwt"]), core.get_max(df_male["Bwt"]))

column_names = df["Sex"].index[0]
print(column_names)

display_hist(df_female)
display_hist(df_male)