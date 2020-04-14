import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
print(df)

#put nan instead of 0 for data where 0 is not possible reasult
df.loc[df["Glucose"] == 0.0, "Glucose"] = np.NAN
df.loc[df["BloodPressure"] == 0.0, "BloodPressure"] = np.NAN
df.loc[df["SkinThickness"] == 0.0, "SkinThickness"] = np.NAN
df.loc[df["Insulin"] == 0.0, "Insulin"] = np.NAN
df.loc[df["BMI"] == 0.0, "BMI"] = np.NAN
print("no of rows: ", df.shape[0])

def calculate_missing_values(df, column):
    percentage = (df[column].isnull().sum()/df.shape[0])*100
    print(f"Percentage of missing values for {column} is {'%.2f' % percentage} %")

for col in df.columns:
    calculate_missing_values(df, col)