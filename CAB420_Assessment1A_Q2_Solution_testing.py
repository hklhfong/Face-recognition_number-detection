import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from scipy.stats import norm


# reading in the binary data set
forest_data = pd.read_csv(r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q2\training.csv')
test = pd.read_csv(r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q2\testing.csv')
X_test = test.drop('class', axis=1)
Y_test = test['class']
# seperating into our covariates/feratures and our response variable
# can get the response variable by just dropping the `quality` column (which is our response variable)
X = forest_data.drop('class', axis=1)
# now get the response variable by just getting the `quality` column
Y = forest_data['class']
# lets separate it into train and test splits as well
# will use 80% for train, 20% for test
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=7)

def eval_number_val(model, X_train, Y_train, X_val, Y_val):
    pred = model.predict(X_val)
    print('Validation Accuracy: ' + str(sum(pred == Y_val)/len(Y_val)))

def eval_model_val(model, X_train, Y_train, X_val, Y_val):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance');
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_val, Y_val, normalize='true', ax=ax)
    conf.ax_.set_title('Validation Set Performance');
    pred = model.predict(X_val)
    print('Validation Accuracy: ' + str(sum(pred == Y_val)/len(Y_val)))

def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance');
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    conf.ax_.set_title('Test Set Performance');
    pred = model.predict(X_test)
    print('Test Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))
  
plt.figure()
plt.hist(Y, 6)
plt.title('Histogram Total data set')
plt.figure()
plt.hist(Y_train, 6)
plt.title('Histogram Training data set')
plt.figure()
plt.hist(Y_val, 6)
plt.title('Histogram Validation data set')    
  
neighbors_num = range(20, 30)
for x in neighbors_num:
    print('KNN in range :' + str(x))
    cknn = KNeighborsClassifier(n_neighbors=x)
    cknn.fit(X_train, Y_train)
    eval_number_val(cknn, X_train, Y_train, X_val, Y_val)
    
    print('KNN Distance in range :' + str(x))
    cknn = KNeighborsClassifier(n_neighbors=x, weights='distance')
    cknn.fit(X_train, Y_train)
    eval_number_val(cknn, X_train, Y_train, X_val, Y_val)





# eval_model(cknn, X_train, Y_train, X_test, Y_test)


# eval_model(cknn, X_train, Y_train, X_test, Y_test)

print('SVC')
svm = SVC()
svm.fit(X_train, Y_train)
eval_model_val(svm, X_train, Y_train, X_val, Y_val)
# eval_model(svm, X_train, Y_train, X_test, Y_test)

print('SVC Balanced')
svm = SVC(class_weight='balanced')
svm.fit(X_train, Y_train)
eval_model_val(svm, X_train, Y_train, X_val, Y_val)
# eval_model(svm, X_train, Y_train, X_test, Y_test)

print('SVC One Vs One Balanced')
onevsone_svm = OneVsOneClassifier(SVC(class_weight='balanced'))
onevsone_svm.fit(X_train, Y_train)
eval_model_val(onevsone_svm, X_train, Y_train, X_val, Y_val)
# eval_model(onevsone_svm, X_train, Y_train, X_test, Y_test)

print('SVC One Vs All Balanced')
onevsall_svm = OneVsRestClassifier(SVC(class_weight='balanced'))
onevsall_svm.fit(X_train, Y_train)
eval_model_val(onevsall_svm, X_train, Y_train, X_val, Y_val)
# eval_model(onevsall_svm, X_train, Y_train, X_test, Y_test)