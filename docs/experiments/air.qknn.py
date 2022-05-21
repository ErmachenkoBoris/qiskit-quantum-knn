#!/usr/bin/env python
# coding: utf-8

# In[1]:

from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
import qiskit as qk
from qiskit.utils import QuantumInstance

import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import figure

# MAIN CONSTS
n_variables = 4        # should be positive power of 2
n_train_points = 16   # can be any positive integer
n_test_points = 16     # can be any positive integer
n_neighbors_local = 5

# DATA PREPARATION

df = pd.read_csv('../../datasets/air.csv', names=['MQ1', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'CO2'])
print(df.head(3))

scaler = StandardScaler()

LABEL_NAME = 'CO2'

X = df.drop([LABEL_NAME], axis = 1)
y = df[LABEL_NAME]

smote = SMOTE('minority')
X, y = smote.fit_resample(X, y)

# Check for balanced
data = y.value_counts().reset_index()
sns.barplot(x='index', y = LABEL_NAME, data=data, palette="cividis");
plt.show()

X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# visualize via PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, y], axis = 1)
PCA_df[LABEL_NAME] = LabelEncoder().fit_transform(PCA_df[LABEL_NAME])
PCA_df.head()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

classes = [3, 2, 1, 0]
colors = ['r', 'b', 'g', 'y']
for clas, color in zip(classes, colors):
   plt.scatter(PCA_df.loc[PCA_df[LABEL_NAME] == clas, 'PC1'],
               PCA_df.loc[PCA_df[LABEL_NAME] == clas, 'PC2'],
               c=color)

plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.title('2D PCA', fontsize=15)
plt.legend(['Normal', 'Prepare', 'Smoke', 'CLEANING'])
plt.grid()
plt.show()

# SEARCHING MOST IMPORTANT FEATURES

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LogisticRegression

#Select top 2 features based on mutual info regression
selector = SelectKBest(mutual_info_regression, k = n_variables)
selector.fit(X, y)
mainSelectKBest = X.columns[selector.get_support()]
print('mainSelectKBest: ', mainSelectKBest)

from sklearn.feature_selection import SequentialFeatureSelector
#Selecting the Best important features according to Logistic Regression
sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select = n_variables, cv =10, direction ='backward')
sfs_selector.fit(X, y)

mainSequentialFeatureSelector = X.columns[sfs_selector.get_support()]
print('mainSequentialFeatureSelector: ', mainSequentialFeatureSelector)

from sklearn.feature_selection import RFE
# #Selecting the Best important features according to Logistic Regression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select = n_variables, step = 1)
rfe_selector.fit(X, y)
mainRFE = X.columns[rfe_selector.get_support()]
print('mainRFE: ', mainRFE)

from sklearn.feature_selection import SelectFromModel
# #Selecting the Best important features according to Logistic Regression using SelectFromModel
sfm_selector = SelectFromModel(estimator=LogisticRegression())
sfm_selector.fit(X, y)
mainSFM = X.columns[sfm_selector.get_support()]
print('mainSFM: ', mainSFM)

from sklearn.feature_selection import SequentialFeatureSelector
#Selecting the Best important features according to Logistic Regression
sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select = n_variables, cv =10, direction ='backward')
sfs_selector.fit(X, y)
mainSFS = X.columns[sfs_selector.get_support()]
print('mainSFS: ', mainSFM)

print('mainFeatures== ', set(mainSFM) & set(mainRFE) & set(mainSelectKBest) & set(mainSequentialFeatureSelector) & set(mainSFS))

mainFeatures = mainSFS.to_numpy()

idf = df[mainFeatures].dropna()

idf = MinMaxScaler().fit_transform(idf)

data_raw = idf

labels = df[LABEL_NAME].to_numpy()

# encode data
encoded_data = analog.encode(data_raw)

# split to train and test
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(encoded_data, labels, test_size = n_test_points,  train_size= n_train_points)

# KNN
start = time.process_time()
modelKNN = KNeighborsClassifier(n_neighbors = n_neighbors_local)
modelKNN.fit(x_training_data, y_training_data)
predictionsKNN = modelKNN.predict(x_test_data)
knnTime = time.process_time() - start

classification_reportKNN = classification_report(y_test_data, predictionsKNN)
accurancyKNN = accuracy_score(y_test_data, predictionsKNN);

# QKNN

# initialising the quantum instance
backend = qk.BasicAer.get_backend('qasm_simulator')

from qiskit.providers.aer import AerSimulator, StatevectorSimulator

simulatorStateVector = StatevectorSimulator()

# Create extended stabilizer method simulator
extended_stabilizer_simulator = AerSimulator(method='extended_stabilizer')

print('count of qubits in simulatorStateVector:', simulatorStateVector.configuration().n_qubits)
print('count of qubits in qasm_simulator:', backend.configuration().n_qubits)
print('count of qubits in extended_stabilizer_simulator:', extended_stabilizer_simulator.configuration().n_qubits)

# Max shots - 65000
instance = QuantumInstance(backend, shots=30000)

# initialising the qknn model
qknn = QKNeighborsClassifier(
   n_neighbors=n_neighbors_local,
   quantum_instance=instance
)

start = time.process_time()
qknn.fit(x_training_data, y_training_data)

qknn_prediction = qknn.predict(x_test_data)
qknnTime = time.process_time() - start

classification_reportKNN = classification_report(y_test_data, qknn_prediction)

print('KNN Time ', knnTime)
print('QKNN Time ', qknnTime)

print('accurency KNN: ', accurancyKNN)
print('accurency QKNN', accuracy_score(y_test_data, qknn_prediction))
