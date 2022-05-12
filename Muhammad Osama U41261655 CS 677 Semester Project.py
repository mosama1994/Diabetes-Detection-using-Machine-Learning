import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn import svm
import seaborn as sns

df_0 = pd.read_csv('Diabetes CDC BRFSS 2015.csv')

# Checking for null values in the dataset
print("Null values in the data set: " + str(df_0.isnull().values.any()))

# Checking size of dataset and if there are sufficient diabetic class vs non diabetic class
print("Dimensions of the dataset: " + str(df_0.shape))
print('Number of people in Diabetes Class: ' + str(len(df_0[df_0['Diabetes_binary'] == 1].index)))
print('Number of people in Non Diabetes Class: ' + str(len(df_0[df_0['Diabetes_binary'] == 0].index)))
print('There are a lot more people in Diabetes class than non Diabetes so while splitting the data we need to check if '
      'the train and testing tests maintain the same or close ratio')

#  Checking the correlation between the features
corrMatrix = df_0.iloc[:, 1:len(df_0.columns)].corr()
sn.heatmap(corrMatrix, annot=True)
plt.savefig("Correlation Matrix.jpg")
plt.show()
print('Looking at the correlation matrix, the features do not show much correlation amongst them with a maximum of '
      '0.52')

# Scaling the data set
scaler = MinMaxScaler(feature_range=(0, 1))
df_0_scaled = scaler.fit_transform(df_0.iloc[:, 1:len(df_0.columns)])
df_1 = pd.DataFrame(df_0_scaled)
df_1.columns = df_0.columns.values[1:len(df_0.columns)]
df_1['Diabetes_binary'] = df_0['Diabetes_binary']

df_2 = df_1[df_1['Diabetes_binary'] == 1].copy(deep=True)
df_3 = df_1[df_1['Diabetes_binary'] == 0].copy(deep=True)
df_4 = df_2.sample(n=5000, replace=False, random_state=10)
df_4.reset_index(inplace=True, drop=True)
df_5 = df_3.sample(n=5000, replace=False, random_state=20)
df_5.reset_index(inplace=True, drop=True)

a_1 = 30


# Function to split the data into training and testing sets:
def train_test_fun(df_6, df_7, b_1):
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df_6.iloc[:, 0:(len(df_6.columns) - 1)].values,
                                                                df_6.iloc[:, len(df_6.columns) - 1].values,
                                                                test_size=0.5, train_size=0.5, random_state=b_1)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df_7.iloc[:, 0:(len(df_7.columns) - 1)].values,
                                                                df_7.iloc[:, len(df_7.columns) - 1].values,
                                                                test_size=0.5, train_size=0.5, random_state=b_1)

    x_train_3 = np.concatenate((x_train_1, x_train_2), axis=0)
    y_train_3 = np.concatenate((y_train_1, y_train_2), axis=0)
    x_test_3 = np.concatenate((x_test_1, x_test_2), axis=0)
    y_test_3 = np.concatenate((y_test_1, y_test_2), axis=0)

    return x_train_3, x_test_3, y_train_3, y_test_3


def remove_feature(weights, df_14, df_15):
    negative = []

    for i_2 in range(0, (df_14.shape[1] - 1)):
        if weights[0, i_2] < 0:
            negative.append(i_2)

    df_8 = df_14.drop(df_14.columns[negative], axis=1)
    df_9 = df_15.drop(df_15.columns[negative], axis=1)

    return df_8, df_9

# KNearest Neighbour Analysis
p_accuracy = [0] * 6
j_1 = 0

for k in [3, 5, 7, 9, 11, 13]:
    x_train, x_test, y_train, y_test = train_test_fun(df_4, df_5, a_1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    p_accuracy[j_1] = accuracy_score(y_test, pred) * 100
    a_1 += 1
    j_1 += 1

k_1 = [3, 5, 7, 9, 11, 13]
plt.plot(k_1, p_accuracy, marker='o')
for x, y in zip(k_1, p_accuracy):
    label = "{:.2f}".format(y)
    plt.annotate(label,  # this is the text
                 (x, y),  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')
plt.xlabel("Number of Neighbours K")
plt.ylabel("Accuracy %")
plt.savefig("KNN Accuracy VS K for P = 2.jpg")
plt.show()
print(
    "\nHighest accuracy for KNN is: " + str(max(p_accuracy)) + "% for " + str(k_1[p_accuracy.index(max(p_accuracy))]) +
    " neighbours & P = 2")

# Logistic Regression
x_train_4, x_test_4, y_train_4, y_test_4 = train_test_fun(df_4, df_5, a_1)
log_res = LogisticRegression()
log_res.fit(x_train_4, y_train_4)
pred_1 = log_res.predict(x_test_4)
log_res_accuracy = accuracy_score(y_test_4, pred_1)
r_score = r2_score(y_test_4, pred_1)
a_1 += 1
print("\nAccuracy for Logistic Regression with all features: " + str(log_res_accuracy * 100) + '%')
log_weights = log_res.coef_
df_10, df_11 = remove_feature(log_weights, df_4, df_5)

x_train_5, x_test_5, y_train_5, y_test_5 = train_test_fun(df_10, df_11, a_1)
log_res_2 = LogisticRegression()
log_res_2.fit(x_train_5, y_train_5)
pred_2 = log_res_2.predict(x_test_5)
log_res_2_accuracy = accuracy_score(y_test_5, pred_2)
a_1 += 1
print("Accuracy for Logistic Regression without features with negative weights: " + str(log_res_2_accuracy * 100) + '%')

# Naive Bayesian
nb = GaussianNB()
x_train_6, x_test_6, y_train_6, y_test_6 = train_test_fun(df_4, df_5, a_1)
nb.fit(x_train_6, y_train_6)
pred_3 = nb.predict(x_test_6)
nb_accuracy = accuracy_score(y_test_6, pred_3)
a_1 += 1
print("\nAccuracy for Naive Bayesian: " + str(nb_accuracy * 100) + '%')

# Linear Discriminant
lda = LDA()
x_train_7, x_test_7, y_train_7, y_test_7 = train_test_fun(df_4, df_5, a_1)
lda.fit(x_train_7, y_train_7)
pred_4 = lda.predict(x_test_7)
lda_accuracy = accuracy_score(y_test_7, pred_4)
a_1 += 1
print("\nAccuracy for Linear Discriminant: " + str(lda_accuracy * 100) + '%')
df_12, df_13 = remove_feature(lda.coef_, df_4, df_5)

x_train_8, x_test_8, y_train_8, y_test_8 = train_test_fun(df_12, df_13, a_1)
lda_2 = LDA()
lda_2.fit(x_train_8, y_train_8)
pred_5 = lda_2.predict(x_test_8)
lda_2_accuracy = accuracy_score(y_test_8, pred_5)
a_1 += 1
print("Accuracy for Linear Discriminant without features with negative weights: " + str(lda_2_accuracy * 100) + '%')

df_16, df_17 = remove_feature(lda_2.coef_, df_12, df_13)

x_train_9, x_test_9, y_train_9, y_test_9 = train_test_fun(df_16, df_17, a_1)
lda_3 = LDA()
lda_3.fit(x_train_9, y_train_9)
pred_6 = lda_3.predict(x_test_9)
lda_3_accuracy = accuracy_score(y_test_9, pred_6)
a_1 += 1
print("Accuracy for Linear Discriminant without features with negative weights: " + str(lda_3_accuracy * 100) + '%')

x_train_10, x_test_10, y_train_10, y_test_10 = train_test_fun(df_4, df_5, a_1)
qda = QDA()
qda.fit(x_train_10, y_train_10)
pred_7 = qda.predict(x_test_10)
qda_accuracy = accuracy_score(y_test_10, pred_7)
a_1 += 1
print("Accuracy for Quadratic Discriminant: " + str(qda_accuracy * 100) + '%')

# Decision Tree Classifier
depth_accuracy = [0] * 28
depth = [i_2 for i_2 in range(1, 29)]

for i_1 in depth:
    x_train_11, x_test_11, y_train_11, y_test_11 = train_test_fun(df_4, df_5, a_1)
    dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=i_1)
    dt.fit(x_train_11, y_train_11)
    pred_8 = dt.predict(x_test_11)
    a_1 += 1
    if i_1 == 7:
        feature_importance = dt.feature_importances_ * 100
    dt_accuracy = accuracy_score(y_test_11, pred_8)
    depth_accuracy[i_1 - 1] = dt_accuracy * 100

plt.plot(depth, depth_accuracy, marker='o')
plt.xlabel("Depth of Decision Tree")
plt.ylabel("Accuracy in %")
plt.savefig("Accuracy VS Depth for Decision Tree Classifier.jpg")
plt.show()

print("\nMax accuracy for Decision Tree is: " + str(max(depth_accuracy)) + "% for depth: " +
      str(depth[depth_accuracy.index(max(depth_accuracy))]))
print("Feature importance for Decision Tree for Max Accuracy: " + str(feature_importance))

# Random Forest Classifier
n_1 = [3, 6, 9, 12, 15, 18, 21]
n_2 = [3, 5, 7, 9, 11, 13, 15]
rmc_accuracy = [[0] * 7, [0] * 7, [0] * 7, [0] * 7, [0] * 7, [0] * 7, [0] * 7]
i_5 = i_6 = 0
max_accuracy = 0
index_depth = 0
index_estimators = 0

for i_3 in n_1:
    for i_4 in n_2:
        x_train_12, x_test_12, y_train_12, y_test_12 = train_test_fun(df_4, df_5, a_1)
        rmc = RandomForestClassifier(n_estimators=i_3, max_depth=i_4, random_state=a_1)
        rmc.fit(x_train_12, y_train_12)
        pred_9 = rmc.predict(x_test_12)
        rmc_accuracy[i_5][i_6] = accuracy_score(y_test_12, pred_9) * 100
        if rmc_accuracy[i_5][i_6] > max_accuracy:
            max_accuracy = rmc_accuracy[i_5][i_6]
            index_depth = i_6
            index_estimators = i_5
            feat_imp = rmc.feature_importances_

        a_1 += 1
        i_6 += 1
    i_5 += 1
    i_6 = 0

plt.plot(n_2, rmc_accuracy[0], label='Number of Estimators = 3', marker='o')
plt.plot(n_2, rmc_accuracy[1], label='Number of Estimators = 6', marker='o')
plt.plot(n_2, rmc_accuracy[2], label='Number of Estimators = 9', marker='o')
plt.plot(n_2, rmc_accuracy[3], label='Number of Estimators = 12', marker='o')
plt.plot(n_2, rmc_accuracy[4], label='Number of Estimators = 15', marker='o')
plt.plot(n_2, rmc_accuracy[5], label='Number of Estimators = 18', marker='o')
plt.plot(n_2, rmc_accuracy[6], label='Number of Estimators = 21', marker='o')

plt.xlabel("Depth of Tree")
plt.ylabel("Percentage Accuracy %")
plt.legend()
plt.savefig("Random Forest Accuracy VS Max Depth & Number of Estimators.jpg")
plt.show()

print("\nMax Accuracy for Random Forest is: " + str(max_accuracy) + "% for " + str(n_1[index_estimators]) +
      " estimators and depth: " + str(n_2[index_depth]))
print("Feature importance for Random Forest for Max Accuracy: " + str(feat_imp) + "\n")

# Adaboost Classifier
log_res_3 = LogisticRegression()
nb_2 = GaussianNB()
rf = RandomForestClassifier(n_estimators=15, max_depth=7, random_state=101)

pred_10 = [[0] * 8, [0] * 8]
ada_accuracy_1 = [[0] * 8, [0] * 8]
max_ada_log = 0
pred_11 = [[0] * 8, [0] * 8]
ada_accuracy_2 = [[0] * 8, [0] * 8]
max_ada_rf = 0
pred_12 = [[0] * 8, [0] * 8]
ada_accuracy_3 = [[0] * 8, [0] * 8]
max_ada_nb = 0
i = j = 0
n = 20

for lambda_1 in [0.5, 1]:
    for N in [1, 3, 5, 7, 9, 11, 13, 15]:
        x_train_13, x_test_13, y_train_13, y_test_13 = train_test_fun(df_4, df_5, a_1)
        model = AdaBoostClassifier(n_estimators=N, base_estimator=log_res_3, learning_rate=lambda_1, random_state=n)
        model.fit(x_train_13, y_train_13)
        pred_10[i][j] = model.predict(x_test_13)
        ada_accuracy_1[i][j] = accuracy_score(y_test_13, pred_10[i][j]) * 100
        if ada_accuracy_1[i][j] > max_ada_log:
            max_ada_log = ada_accuracy_1[i][j]
            lambda_log = lambda_1
            N_log = N
        a_1 += 1

        x_train_14, x_test_14, y_train_14, y_test_14 = train_test_fun(df_4, df_5, a_1)
        model_1 = AdaBoostClassifier(n_estimators=N, base_estimator=rf, learning_rate=lambda_1,
                                     random_state=n)
        model_1.fit(x_train_14, y_train_14)
        pred_11[i][j] = model_1.predict(x_test_14)
        ada_accuracy_2[i][j] = accuracy_score(y_test_14, pred_11[i][j]) * 100
        if ada_accuracy_2[i][j] > max_ada_rf:
            max_ada_rf = ada_accuracy_2[i][j]
            lambda_rf = lambda_1
            N_rf = N
        a_1 += 1

        x_train_15, x_test_15, y_train_15, y_test_15 = train_test_fun(df_4, df_5, a_1)
        model_2 = AdaBoostClassifier(n_estimators=N, base_estimator=nb_2, learning_rate=lambda_1, random_state=n)
        model_2.fit(x_train_15, y_train_15)
        pred_12[i][j] = model_2.predict(x_test_15)
        ada_accuracy_3[i][j] = accuracy_score(y_test_15, pred_12[i][j]) * 100
        if ada_accuracy_3[i][j] > max_ada_nb:
            max_ada_nb = ada_accuracy_3[i][j]
            lambda_nb_2 = lambda_1
            N_nb_2 = N
        a_1 += 1

        n += 1
        j += 1
    j = 0
    i += 1

N = [1, 3, 5, 7, 9, 11, 13, 15]
plt.plot(N, ada_accuracy_1[0], marker='o', label='lambda = 0.5')
plt.plot(N, ada_accuracy_1[1], marker='o', label='lambda = 1')
plt.legend()
plt.ylabel('Accuracy %')
plt.xlabel('N')
plt.title('Base Estimator = Logistic Regression')
plt.savefig("AdaBoost with Base Estimator = Logistic Regression.jpg")
plt.show()

plt.plot(N, ada_accuracy_2[0], marker='o', label='lambda = 0.5')
plt.plot(N, ada_accuracy_2[1], marker='o', label='lambda = 1')
plt.legend()
plt.ylabel('Accuracy %')
plt.xlabel('N')
plt.title('Base Estimator = Random Forest Classifier')
plt.savefig("AdaBoost with Base Estimator = Random Forest Classifier.jpg")
plt.show()

plt.plot(N, ada_accuracy_3[0], marker='o', label='lambda = 0.5')
plt.plot(N, ada_accuracy_3[1], marker='o', label='lambda = 1')
plt.legend()
plt.ylabel('Accuracy %')
plt.xlabel('N')
plt.title('Base Estimator = Naive Bayesian')
plt.savefig("AdaBoost with Base Estimator = Naive Bayesian.jpg")
plt.show()

print("Highest Accuracy for Adaboost with Logistic Regression is: " + str(max_ada_log) + "% with Learning rate: "
      + str(lambda_log) + " & " + str(N_log) + " base estimators")
print("Highest Accuracy for Adaboost with Random Forest is: " + str(max_ada_rf) + "% with Learning rate: "
      + str(lambda_rf) + " & " + str(N_rf) + " base estimators")
print("Highest Accuracy for Adaboost with Naive Bayesian is: " + str(max_ada_nb) + "% with Learning rate: "
      + str(lambda_nb_2) + " & " + str(N_nb_2) + " base estimators")

# K-Means Clustering
frames = [df_4, df_5]
df_18 = pd.concat(frames)
x_1 = df_18.iloc[:, 0:21].values

k_2 = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
inertia = []

for i in k_2:
    km = KMeans(n_clusters=i, random_state=i)
    pred_13 = km.fit_predict(x_1)
    inertia.append(km.inertia_)
    centroids = km.cluster_centers_

plt.plot(k_2, inertia, marker='o', color='green')
plt.xlabel('Number of clusters: K')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig("Inertia VS Number of clusters.jpg")
plt.show()

# SVM
x_train_16, x_test_16, y_train_16, y_test_16 = train_test_fun(df_4, df_5, a_1)
svm_linear = svm.SVC(kernel="linear")
svm_linear.fit(x_train_16, y_train_16)
pred_14 = svm_linear.predict(x_test_16)
svm_linear_accuracy = accuracy_score(y_test_16, pred_14)
a_1 += 1
print("\nAccuracy for Linear SVM is: " + str(svm_linear_accuracy * 100) + "%")

x_train_17, x_test_17, y_train_17, y_test_17 = train_test_fun(df_4, df_5, a_1)
svm_gaussian = svm.SVC(kernel="rbf")
svm_gaussian.fit(x_train_17, y_train_17)
pred_15 = svm_gaussian.predict(x_test_17)
svm_gaussian_accuracy = accuracy_score(y_test_17, pred_15)
a_1 += 1
print("Accuracy for Gaussian SVM is: " + str(svm_gaussian_accuracy * 100) + "%")

for i_7 in [2, 3, 4, 5]:
    x_train_18, x_test_18, y_train_18, y_test_18 = train_test_fun(df_4, df_5, a_1)
    svm_poly = svm.SVC(kernel="poly", degree=i_7)
    svm_poly.fit(x_train_18, y_train_18)
    pred_16 = svm_poly.predict(x_test_18)
    svm_poly_accuracy = accuracy_score(y_test_18, pred_16)
    a_1 += 1
    print("Accuracy for Polynomial SVM of degree " + str(i_7) + " is: " + str(svm_gaussian_accuracy * 100) + "%")
