import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

####https://www.kaggle.com/pierra/credit-card-dataset-svm-classification/comments#233689####


df = pd.read_csv('dataset/creditcard.csv', sep=",", header=0, encoding= 'unicode_escape')
df = df.dropna(how='all')
print(df)

fraud = len(df[df['Class'] == 1])
no_fraud = len(df[df['Class'] == 0])
print("There is ", fraud, " data in the dataset and ", no_fraud," data")

# Separate majority and minority classes
df_majority = df[df['Class']==0]
df_minority = df[df['Class']==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=fraud,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled['Class'].value_counts()


X = df_downsampled.drop(['Time', 'Class'],axis=1) # drop Time (useless), and the Class (label)
y = df_downsampled['Class'] #create label

# X = X.head(50000)
# y = y.head(50000)

# weight_fraud = len(y[y == 1])/len(y)
# print(weight_fraud )
# weight_no_fraud = len(y[y == 0])/len(y)
# class_weights = {0: weight_no_fraud, 1: weight_fraud}

# print(class_weights)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


print(X_train)
print(y_train)
svclassifier = SVC(kernel='linear', C=1)
scores = cross_val_score(svclassifier, X_train, y_train, cv=10)

plt.plot(scores)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title("SVM model accuracy over iterations")


print("scores ", scores)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['component 1', 'component 2'])

print(principalDf)

#finalDf = pd.concat([principalDf, y], axis = 1)


# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('component 1', fontsize = 15)
# ax.set_ylabel('component 2', fontsize = 15)
# ax.set_title('PCA', fontsize = 20)
# targets = [1, 0]
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Class'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'component 1'], finalDf.loc[indicesToKeep, 'component 2'], c = color, s = 50)
# ax.legend(targets)
# ax.grid()
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of the Dataset",fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(principalDf.loc[indicesToKeep, 'component 1'], principalDf.loc[indicesToKeep, 'component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

plt.show()
history = svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

# model accuracy for X_test  
accuracy = svclassifier.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(accuracy)

#get correct predictions
correct_predictions = []
for i in range(len(X_test)):
	if y_pred[i] == y_test[i]:
		correct_predictions.append(y_pred[i])

print("Accuracy of the model is ", len(correct_predictions)/len(X_test))


