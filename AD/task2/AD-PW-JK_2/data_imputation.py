import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('led.csv', sep=",")
df = df[df['Lifeexpectancy'].notnull()]

print(df)


def calculate_missing_values(df, column):
    percentage = (df[column].isnull().sum()/df.shape[0])*100
    print(f"Percentage of missing values for {column} is {'%.2f' % percentage} %")

def data_information(df):
    print("Data information:") 
    print(df.describe().loc[['mean', 'std', '25%', '50%', '75%']].astype(float).applymap('{:,.2f}'.format))

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 

df_num_cols = pd.DataFrame(df[["AdultMortality", "infantdeaths", "Alcohol", "percentageexpenditure", "HepatitisB", "BMI", "under-fivedeaths", "Totalexpenditure", "GDP", "Population"]])
missing_columns = pd.DataFrame(df[["Alcohol", "HepatitisB", "BMI", "Totalexpenditure", "GDP", "Population"]])
non_missing_columns = df_num_cols[~df_num_cols.isin(missing_columns)].dropna(axis='columns', how="all")
#non_missing_columns = non_missing_columns[non_missing_columns['_merge']=='left_only']

for col in df_num_cols.columns:
    calculate_missing_values(df_num_cols, col)

print("Non missing cols: ",non_missing_columns)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.2)
fig.suptitle('Influance of chosen factors on Life Expectancy')

for col, ax in zip(df_num_cols, axes.flatten()):
    df_not_null = df[[col, 'Lifeexpectancy']].dropna()
    y_data = df_not_null[[col]].values.reshape(-1, 1)
    x_data = df_not_null[['Lifeexpectancy']].values.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x_data, y_data)  # perform linear regression

    Y_pred = linear_regressor.predict(x_data)  # make predictions
    print('y-PRED: ', Y_pred)
    print(f"Linear regression parameters for {col}")
    # The coefficients
    print('Coefficients: \n', linear_regressor.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
    % np.mean((linear_regressor.predict(x_data) -y_data) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linear_regressor.score(x_data, y_data))
    ax.scatter(x_data, y_data)
    ax.plot(x_data, Y_pred, color='red')
    ax.set(title=col)

  
missing_columns_imputation_mean = pd.DataFrame(missing_columns.fillna(missing_columns.mean()))
#missing_columns_imputation_interpol = missing_columns.interpolate(method ='linear', limit_direction ='forward') 
#missing_columns_imputation_hotdeck = missing_columns.fillna(method='ffill')
#missing_columns_imputation_regression = pd.DataFrame(missing_columns)

# for feature in missing_columns:
#     print(feature)
#     df_not_null = df[[feature, 'Lifeexpectancy']].dropna()
#     y_data = df_not_null[[feature]].values.reshape(-1, 1)
#     x_data = df_not_null[['Lifeexpectancy']].values.reshape(-1, 1)
#     linear_regressor = LinearRegression()  # create object for the class
#     linear_regressor.fit(x_data, y_data)  # perform linear regression
#     missing_columns_imputation_regression.loc[missing_columns_imputation_regression[feature].isnull(), feature] = linear_regressor.predict(df[['Lifeexpectancy']].values.reshape(-1,1))[missing_columns_imputation_regression[feature].isnull().values.reshape(-1,1)]


for col in missing_columns_imputation_mean.columns:
    calculate_missing_values(missing_columns_imputation_mean, col)

fig_before, axes_before = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
fig_before.subplots_adjust(hspace=0.5, wspace=0.2)
fig_before.suptitle('Influance of chosen factors on Life Expectancy before imputation')

for col, ax in zip(missing_columns, axes_before.flatten()):
    df_not_null = df[[col, 'Lifeexpectancy']].dropna()
    x_data = df_not_null[[col]].values.reshape(-1, 1)
    y_data = df_not_null[['Lifeexpectancy']].values.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x_data, y_data)  # perform linear regression
    Y_pred = linear_regressor.predict(x_data)  # make predictions
    print(f"Linear regression parameters for {col}")
    # The coefficients
    print('Coefficients: \n', linear_regressor.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
    % np.mean((linear_regressor.predict(x_data) -y_data) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linear_regressor.score(x_data, y_data))
    ax.scatter(x_data, y_data)
    ax.plot(x_data, Y_pred, color='red')
    ax.set(title=col)

fig_after, axes_after = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
fig_after.subplots_adjust(hspace=0.5, wspace=0.2)
fig_after.suptitle('Influance of chosen factors on Life Expectancy after imputation with mean')

print("Linear regression parameters for data after imputation with mean:")
for col, ax in zip(missing_columns_imputation_mean, axes_after.flatten()):
    x_data = missing_columns_imputation_mean[[col]].values.reshape(-1, 1)
    y_data = df[['Lifeexpectancy']].values.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x_data, y_data)  # perform linear regression
    Y_pred = linear_regressor.predict(x_data)  # make predictions
    print(f"Linear regression parameters for {col}")
    # The coefficients
    print('Coefficients: \n', linear_regressor.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
    % np.mean((linear_regressor.predict(x_data) -y_data) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % linear_regressor.score(x_data, y_data))
    ax.scatter(x_data, y_data)
    ax.plot(x_data, Y_pred, color='red')
    ax.set(title=col)


print("Data before impiutation")
data_information(missing_columns)
print("Data after impiutation with mean")
data_information(missing_columns_imputation_mean)

plt.show()